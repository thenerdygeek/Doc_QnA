import { renderHook, act, waitFor } from "@testing-library/react";
import { useSettings } from "./use-settings";
import { vi } from "vitest";
import type {
  ConfigData,
  ConfigUpdateResponse,
  DbTestResponse,
  DbMigrateResponse,
} from "@/types/api";

vi.mock("@/api/client", () => ({
  api: {
    config: {
      get: vi.fn(),
      update: vi.fn(),
      dbTest: vi.fn(),
      dbMigrate: vi.fn(),
    },
  },
}));

import { api } from "@/api/client";

const mockConfigGet = vi.mocked(api.config.get);
const mockConfigUpdate = vi.mocked(api.config.update);
const mockDbTest = vi.mocked(api.config.dbTest);
const mockDbMigrate = vi.mocked(api.config.dbMigrate);

const FAKE_CONFIG: ConfigData = {
  llm: { provider: "cody", model: "claude-3" },
  database: { url: "postgresql://localhost/test" },
};

const FAKE_UPDATE_RESPONSE: ConfigUpdateResponse = {
  saved: true,
  restart_required: true,
  restart_sections: ["llm"],
};

beforeEach(() => {
  vi.clearAllMocks();
  mockConfigGet.mockResolvedValue(FAKE_CONFIG);
});

describe("useSettings", () => {
  it("config is null before dialog opens, loading=false", () => {
    const { result } = renderHook(() => useSettings());
    expect(result.current.config).toBeNull();
    expect(result.current.loading).toBe(false);
  });

  it("opening dialog triggers GET /api/config via api.config.get()", async () => {
    const { result } = renderHook(() => useSettings());

    act(() => {
      result.current.setOpen(true);
    });

    await waitFor(() => {
      expect(mockConfigGet).toHaveBeenCalledTimes(1);
    });
  });

  it("loading is true during fetch, false after", async () => {
    let resolveGet!: (value: ConfigData) => void;
    mockConfigGet.mockReturnValue(
      new Promise((resolve) => {
        resolveGet = resolve;
      }),
    );

    const { result } = renderHook(() => useSettings());

    act(() => {
      result.current.setOpen(true);
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(true);
    });

    await act(async () => {
      resolveGet(FAKE_CONFIG);
    });

    expect(result.current.loading).toBe(false);
  });

  it("config populated after fetch resolves", async () => {
    const { result } = renderHook(() => useSettings());

    act(() => {
      result.current.setOpen(true);
    });

    await waitFor(() => {
      expect(result.current.config).toEqual(FAKE_CONFIG);
    });
  });

  it("second setOpen(true) does not re-fetch (hasFetched ref)", async () => {
    const { result } = renderHook(() => useSettings());

    act(() => {
      result.current.setOpen(true);
    });

    await waitFor(() => {
      expect(result.current.config).toEqual(FAKE_CONFIG);
    });

    // Close then re-open
    act(() => {
      result.current.setOpen(false);
    });
    act(() => {
      result.current.setOpen(true);
    });

    // Still only one call
    expect(mockConfigGet).toHaveBeenCalledTimes(1);
  });

  it("updateSection sends PATCH with correct payload via api.config.update()", async () => {
    mockConfigUpdate.mockResolvedValue({
      saved: true,
      restart_required: false,
      restart_sections: [],
    });

    const { result } = renderHook(() => useSettings());

    // Open to trigger initial fetch
    act(() => {
      result.current.setOpen(true);
    });
    await waitFor(() => {
      expect(result.current.config).toEqual(FAKE_CONFIG);
    });

    await act(async () => {
      await result.current.updateSection("llm", { model: "gpt-4" });
    });

    expect(mockConfigUpdate).toHaveBeenCalledWith({
      llm: { model: "gpt-4" },
    });
  });

  it("after updateSection, config re-fetched via api.config.get()", async () => {
    const updatedConfig: ConfigData = {
      ...FAKE_CONFIG,
      llm: { provider: "cody", model: "gpt-4" },
    };
    mockConfigUpdate.mockResolvedValue({
      saved: true,
      restart_required: false,
      restart_sections: [],
    });
    // First call returns original, second returns updated
    mockConfigGet.mockResolvedValueOnce(FAKE_CONFIG).mockResolvedValueOnce(updatedConfig);

    const { result } = renderHook(() => useSettings());

    act(() => {
      result.current.setOpen(true);
    });
    await waitFor(() => {
      expect(result.current.config).toEqual(FAKE_CONFIG);
    });

    await act(async () => {
      await result.current.updateSection("llm", { model: "gpt-4" });
    });

    // config.get called twice: once on open, once after update
    expect(mockConfigGet).toHaveBeenCalledTimes(2);
    expect(result.current.config).toEqual(updatedConfig);
  });

  it("restart_sections accumulated across saves via Set", async () => {
    mockConfigUpdate
      .mockResolvedValueOnce({
        saved: true,
        restart_required: true,
        restart_sections: ["llm"],
      })
      .mockResolvedValueOnce({
        saved: true,
        restart_required: true,
        restart_sections: ["database", "llm"],
      });

    const { result } = renderHook(() => useSettings());

    act(() => {
      result.current.setOpen(true);
    });
    await waitFor(() => {
      expect(result.current.config).not.toBeNull();
    });

    await act(async () => {
      await result.current.updateSection("llm", { model: "gpt-4" });
    });
    expect(result.current.restartRequired).toEqual(["llm"]);

    await act(async () => {
      await result.current.updateSection("database", { url: "pg://new" });
    });

    // Set deduplicates "llm"
    expect(result.current.restartRequired).toContain("llm");
    expect(result.current.restartRequired).toContain("database");
    expect(result.current.restartRequired.length).toBe(2);
  });

  it("saving is true during updateSection, false after", async () => {
    let resolveUpdate!: (value: ConfigUpdateResponse) => void;
    mockConfigUpdate.mockReturnValue(
      new Promise((resolve) => {
        resolveUpdate = resolve;
      }),
    );

    const { result } = renderHook(() => useSettings());

    act(() => {
      result.current.setOpen(true);
    });
    await waitFor(() => {
      expect(result.current.config).not.toBeNull();
    });

    let updatePromise: Promise<string[]>;
    act(() => {
      updatePromise = result.current.updateSection("llm", { model: "x" });
    });

    await waitFor(() => {
      expect(result.current.saving).toBe(true);
    });

    await act(async () => {
      resolveUpdate({
        saved: true,
        restart_required: false,
        restart_sections: [],
      });
      await updatePromise;
    });

    expect(result.current.saving).toBe(false);
  });

  it("testDbConnection calls api.config.dbTest with url", async () => {
    const dbResult: DbTestResponse = { ok: true };
    mockDbTest.mockResolvedValue(dbResult);

    const { result } = renderHook(() => useSettings());

    await act(async () => {
      await result.current.testDbConnection("postgresql://localhost/db");
    });

    expect(mockDbTest).toHaveBeenCalledWith("postgresql://localhost/db");
  });

  it("DB test success: dbTestResult set to {ok: true}", async () => {
    mockDbTest.mockResolvedValue({ ok: true });

    const { result } = renderHook(() => useSettings());

    await act(async () => {
      await result.current.testDbConnection("postgresql://localhost/db");
    });

    expect(result.current.dbTestResult).toEqual({ ok: true });
  });

  it("DB test failure: dbTestResult set to {ok: false, error: '...'}", async () => {
    mockDbTest.mockResolvedValue({ ok: false, error: "Connection refused" });

    const { result } = renderHook(() => useSettings());

    await act(async () => {
      await result.current.testDbConnection("bad-url");
    });

    expect(result.current.dbTestResult).toEqual({
      ok: false,
      error: "Connection refused",
    });
  });

  it("runMigrations calls api.config.dbMigrate", async () => {
    const migrateResult: DbMigrateResponse = { ok: true, revision: "abc123" };
    mockDbMigrate.mockResolvedValue(migrateResult);

    const { result } = renderHook(() => useSettings());

    await act(async () => {
      await result.current.runMigrations();
    });

    expect(mockDbMigrate).toHaveBeenCalledTimes(1);
  });

  it("migrate result stored", async () => {
    const migrateResult: DbMigrateResponse = { ok: true, revision: "abc123" };
    mockDbMigrate.mockResolvedValue(migrateResult);

    const { result } = renderHook(() => useSettings());

    await act(async () => {
      await result.current.runMigrations();
    });

    expect(result.current.migrateResult).toEqual(migrateResult);
  });

  it("previous dbTestResult cleared before new test", async () => {
    mockDbTest
      .mockResolvedValueOnce({ ok: true })
      .mockImplementation(
        () => new Promise(() => {}), // second call never resolves
      );

    const { result } = renderHook(() => useSettings());

    await act(async () => {
      await result.current.testDbConnection("url1");
    });
    expect(result.current.dbTestResult).toEqual({ ok: true });

    // Start a second test (will not resolve)
    act(() => {
      void result.current.testDbConnection("url2");
    });

    // Result should be cleared immediately
    await waitFor(() => {
      expect(result.current.dbTestResult).toBeNull();
    });
  });

  it("config fetch failure does not crash (stays null)", async () => {
    mockConfigGet.mockRejectedValue(new Error("Server down"));
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    const { result } = renderHook(() => useSettings());

    act(() => {
      result.current.setOpen(true);
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.config).toBeNull();
    consoleSpy.mockRestore();
  });
});

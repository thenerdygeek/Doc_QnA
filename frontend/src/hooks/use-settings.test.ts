import { renderHook, act, waitFor } from "@testing-library/react";
import { useSettings } from "./use-settings";
import { vi } from "vitest";
import type {
  ConfigData,
  ConfigUpdateResponse,
} from "@/types/api";

vi.mock("@/api/client", () => ({
  api: {
    config: {
      get: vi.fn(),
      update: vi.fn(),
    },
  },
}));

import { api } from "@/api/client";

const mockConfigGet = vi.mocked(api.config.get);
const mockConfigUpdate = vi.mocked(api.config.update);

const FAKE_CONFIG: ConfigData = {
  llm: { provider: "cody", model: "claude-3" },
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
        restart_sections: ["indexing", "llm"],
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
      await result.current.updateSection("indexing", { chunk_size: 256 });
    });

    // Set deduplicates "llm"
    expect(result.current.restartRequired).toContain("llm");
    expect(result.current.restartRequired).toContain("indexing");
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

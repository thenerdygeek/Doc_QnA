import { renderHook, act } from "@testing-library/react";
import { useTour, TOUR_STEPS } from "./use-tour";

const TOUR_COMPLETED_KEY = "doc-qa-tour-completed";

beforeEach(() => {
  localStorage.clear();
});

describe("useTour", () => {
  it("first visit: isFirstVisit=true, tour auto-starts (active=true, currentStep=0)", () => {
    const { result } = renderHook(() => useTour());
    expect(result.current.isFirstVisit).toBe(true);
    expect(result.current.active).toBe(true);
    expect(result.current.currentStep).toBe(0);
  });

  it("return visit (localStorage key set): active=false, isFirstVisit=false", () => {
    localStorage.setItem(TOUR_COMPLETED_KEY, "true");
    const { result } = renderHook(() => useTour());
    expect(result.current.isFirstVisit).toBe(false);
    expect(result.current.active).toBe(false);
  });

  it("next() advances step", () => {
    const { result } = renderHook(() => useTour());
    expect(result.current.currentStep).toBe(0);

    act(() => {
      result.current.next();
    });

    expect(result.current.currentStep).toBe(1);
  });

  it("next() at last step returns false", () => {
    const { result } = renderHook(() => useTour());

    // Advance to last step
    for (let i = 0; i < TOUR_STEPS.length - 1; i++) {
      act(() => {
        result.current.next();
      });
    }
    expect(result.current.currentStep).toBe(TOUR_STEPS.length - 1);

    let returned: boolean = true;
    act(() => {
      returned = result.current.next();
    });

    expect(returned).toBe(false);
    expect(result.current.currentStep).toBe(TOUR_STEPS.length - 1);
  });

  it("back() decrements step", () => {
    const { result } = renderHook(() => useTour());

    // Go forward first
    act(() => {
      result.current.next();
    });
    act(() => {
      result.current.next();
    });
    expect(result.current.currentStep).toBe(2);

    act(() => {
      result.current.back();
    });
    expect(result.current.currentStep).toBe(1);
  });

  it("back() at step 0 stays at 0", () => {
    const { result } = renderHook(() => useTour());
    expect(result.current.currentStep).toBe(0);

    act(() => {
      result.current.back();
    });

    expect(result.current.currentStep).toBe(0);
  });

  it("skip() advances step", () => {
    const { result } = renderHook(() => useTour());
    expect(result.current.currentStep).toBe(0);

    act(() => {
      result.current.skip();
    });

    expect(result.current.currentStep).toBe(1);
  });

  it("finish() deactivates, resets to 0, writes localStorage", () => {
    const { result } = renderHook(() => useTour());

    // Advance a couple steps first
    act(() => {
      result.current.next();
    });
    act(() => {
      result.current.next();
    });
    expect(result.current.currentStep).toBe(2);

    act(() => {
      result.current.finish();
    });

    expect(result.current.active).toBe(false);
    expect(result.current.currentStep).toBe(0);
    expect(localStorage.getItem(TOUR_COMPLETED_KEY)).toBe("true");
  });

  it("start() reactivates from step 0", () => {
    localStorage.setItem(TOUR_COMPLETED_KEY, "true");
    const { result } = renderHook(() => useTour());
    expect(result.current.active).toBe(false);

    act(() => {
      result.current.start();
    });

    expect(result.current.active).toBe(true);
    expect(result.current.currentStep).toBe(0);
  });

  it("each step has correct tab reference", () => {
    const { result } = renderHook(() => useTour());

    for (let i = 0; i < TOUR_STEPS.length; i++) {
      expect(TOUR_STEPS[i].tab).toBe(TOUR_STEPS[i].tab);
      // Verify step data matches
      if (i === result.current.currentStep) {
        expect(result.current.step.tab).toBe(TOUR_STEPS[i].tab);
      }
    }

    // Verify specific known tabs
    expect(TOUR_STEPS[0].tab).toBeNull(); // welcome
    expect(TOUR_STEPS[1].tab).toBe("llm");
    expect(TOUR_STEPS[2].tab).toBe("database");
    expect(TOUR_STEPS[3].tab).toBe("retrieval");
    expect(TOUR_STEPS[4].tab).toBeNull(); // advanced
    expect(TOUR_STEPS[5].tab).toBeNull(); // complete
  });

  it("totalSteps === 6", () => {
    const { result } = renderHook(() => useTour());
    expect(result.current.totalSteps).toBe(6);
  });

  it("step object matches TOUR_STEPS", () => {
    const { result } = renderHook(() => useTour());

    // Verify step at index 0
    expect(result.current.step).toEqual(TOUR_STEPS[0]);

    // Advance and verify step at index 1
    act(() => {
      result.current.next();
    });
    expect(result.current.step).toEqual(TOUR_STEPS[1]);

    // Advance and verify step at index 2
    act(() => {
      result.current.next();
    });
    expect(result.current.step).toEqual(TOUR_STEPS[2]);
  });
});

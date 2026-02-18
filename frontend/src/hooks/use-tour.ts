import { useCallback, useEffect, useState } from "react";

const TOUR_COMPLETED_KEY = "doc-qa-tour-completed";

export interface TourStep {
  id: string;
  /** Tab to activate in the settings dialog, or null for non-tab steps. */
  tab: string | null;
  title: string;
  description: string;
  required: boolean;
}

export const TOUR_STEPS: TourStep[] = [
  {
    id: "welcome",
    tab: null,
    title: "Welcome to Doc QA",
    description:
      "Let's walk through the key settings so you can get the most out of your documentation Q&A system. This will only take a minute.",
    required: true,
  },
  {
    id: "llm",
    tab: "llm",
    title: "AI Backend",
    description:
      "This is where you configure which LLM powers your answers. Cody is pre-configured by default. Verify the model and endpoint, or switch to Ollama for local inference.",
    required: true,
  },
  {
    id: "retrieval",
    tab: "retrieval",
    title: "Search Settings",
    description:
      "Control how documents are searched and ranked. The defaults work well for most use cases — adjust top_k, min_score, or toggle reranking if needed.",
    required: false,
  },
  {
    id: "advanced",
    tab: null,
    title: "Advanced Settings",
    description:
      "The remaining tabs — Intelligence, Generation, Verification, and Indexing — offer fine-grained control over intent classification, diagram generation, CRAG verification, and chunk sizes. Explore them anytime from the gear icon.",
    required: false,
  },
  {
    id: "complete",
    tab: null,
    title: "You're all set!",
    description:
      "Close this dialog and start asking questions. Your indexed documentation is ready to go. You can always re-open Settings or restart this tour from the settings footer.",
    required: true,
  },
];

export interface UseTourReturn {
  /** Whether the tour is currently active. */
  active: boolean;
  /** Current step index (0-based). */
  currentStep: number;
  /** The current step definition. */
  step: TourStep;
  /** Total number of steps. */
  totalSteps: number;
  /** Navigate to the next step. Returns false if already at the end. */
  next: () => boolean;
  /** Navigate to the previous step. */
  back: () => void;
  /** Skip the current step (same as next for optional steps). */
  skip: () => void;
  /** Start the tour from the beginning. */
  start: () => void;
  /** End the tour and mark as completed. */
  finish: () => void;
  /** Whether this is the user's first visit (tour never completed). */
  isFirstVisit: boolean;
}

export function useTour(): UseTourReturn {
  const [active, setActive] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [isFirstVisit] = useState(() => {
    return localStorage.getItem(TOUR_COMPLETED_KEY) === null;
  });

  // Auto-start tour on first visit
  useEffect(() => {
    if (isFirstVisit) {
      setActive(true);
      setCurrentStep(0);
    }
  }, [isFirstVisit]);

  // currentStep is always in [0, TOUR_STEPS.length - 1]
  const step = TOUR_STEPS[currentStep]!;

  const next = useCallback((): boolean => {
    if (currentStep >= TOUR_STEPS.length - 1) {
      return false;
    }
    setCurrentStep((s) => s + 1);
    return true;
  }, [currentStep]);

  const back = useCallback(() => {
    setCurrentStep((s) => Math.max(0, s - 1));
  }, []);

  const skip = useCallback(() => {
    // Skip behaves like next
    if (currentStep < TOUR_STEPS.length - 1) {
      setCurrentStep((s) => s + 1);
    }
  }, [currentStep]);

  const start = useCallback(() => {
    setCurrentStep(0);
    setActive(true);
  }, []);

  const finish = useCallback(() => {
    setActive(false);
    setCurrentStep(0);
    localStorage.setItem(TOUR_COMPLETED_KEY, "true");
  }, []);

  return {
    active,
    currentStep,
    step,
    totalSteps: TOUR_STEPS.length,
    next,
    back,
    skip,
    start,
    finish,
    isFirstVisit,
  };
}

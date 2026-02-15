import {
    ChevronRight,
    ChevronLeft,
    SkipForward,
    PartyPopper,
    Rocket,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { UseTourReturn } from "@/hooks/use-tour";

export function TourOverlay({ tour }: { tour: UseTourReturn }) {
    const { step, currentStep, totalSteps, next, back, skip, finish } = tour;
    const isFirst = currentStep === 0;
    const isLast = currentStep === totalSteps - 1;

    return (
        <div className="rounded-lg border border-primary/30 bg-primary/5 p-4 dark:bg-primary/10">
            {/* Progress bar */}
            <div className="mb-3 flex items-center gap-2">
                <div className="flex-1">
                    <div className="bg-muted h-1.5 rounded-full overflow-hidden">
                        <div
                            className="bg-primary h-full rounded-full transition-all duration-300"
                            style={{ width: `${((currentStep + 1) / totalSteps) * 100}%` }}
                        />
                    </div>
                </div>
                <span className="text-muted-foreground text-[10px] font-medium tabular-nums">
                    {currentStep + 1}/{totalSteps}
                </span>
            </div>

            {/* Step content */}
            <div className="mb-3 flex items-start gap-2">
                {isLast ? (
                    <PartyPopper className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                ) : isFirst ? (
                    <Rocket className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                ) : null}
                <div>
                    <div className="flex items-center gap-2">
                        <h4 className="text-sm font-semibold">{step.title}</h4>
                        {step.required ? (
                            <Badge variant="default" className="h-4 px-1.5 text-[9px]">
                                Required
                            </Badge>
                        ) : (
                            <Badge variant="secondary" className="h-4 px-1.5 text-[9px]">
                                Optional
                            </Badge>
                        )}
                    </div>
                    <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                        {step.description}
                    </p>
                </div>
            </div>

            {/* Navigation */}
            <div className="flex items-center justify-between">
                <div>
                    {!isFirst && (
                        <Button variant="ghost" size="xs" onClick={back}>
                            <ChevronLeft className="mr-0.5 h-3 w-3" /> Back
                        </Button>
                    )}
                </div>
                <div className="flex items-center gap-2">
                    {!step.required && !isLast && (
                        <Button variant="ghost" size="xs" onClick={skip} className="text-muted-foreground">
                            <SkipForward className="mr-0.5 h-3 w-3" /> Skip
                        </Button>
                    )}
                    {isLast ? (
                        <Button size="xs" onClick={finish}>
                            Get Started <ChevronRight className="ml-0.5 h-3 w-3" />
                        </Button>
                    ) : (
                        <Button size="xs" onClick={() => next()}>
                            Next <ChevronRight className="ml-0.5 h-3 w-3" />
                        </Button>
                    )}
                </div>
            </div>
        </div>
    );
}

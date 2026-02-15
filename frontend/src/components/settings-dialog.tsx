import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Loader2,
  Database,
  Search,
  Brain,
  Sparkles,
  ShieldCheck,
  Radio,
  RotateCcw,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { UseSettingsReturn } from "@/hooks/use-settings";
import type { UseTourReturn } from "@/hooks/use-tour";
import type { UseIndexingReturn } from "@/hooks/use-indexing";
import { TourOverlay } from "@/components/settings/tour-overlay";
import { DatabaseTab } from "@/components/settings/database-tab";
import { RetrievalTab } from "@/components/settings/retrieval-tab";
import { LLMTab } from "@/components/settings/llm-tab";
import { IntelligenceTab } from "@/components/settings/intelligence-tab";
import { GenerationTab } from "@/components/settings/generation-tab";
import { VerificationTab } from "@/components/settings/verification-tab";
import { IndexingTab } from "@/components/settings/indexing-tab";

// ── Main Dialog ─────────────────────────────────────────────────

interface SettingsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  settings: UseSettingsReturn;
  tour: UseTourReturn;
  onDbSaved?: () => void;
  indexing: UseIndexingReturn;
}

export function SettingsDialog({ open, onOpenChange, settings, tour, onDbSaved, indexing }: SettingsDialogProps) {
  const { loading, restartRequired } = settings;
  const [activeTab, setActiveTab] = useState("database");

  // When tour is active and step has a tab, switch to it
  useEffect(() => {
    if (tour.active && tour.step.tab) {
      setActiveTab(tour.step.tab);
    }
  }, [tour.active, tour.step]);

  // After DB save + migrate, notify parent so it can refresh conversations
  const originalUpdateSection = settings.updateSection;
  const wrappedSettings: UseSettingsReturn = {
    ...settings,
    updateSection: async (section, data) => {
      const result = await originalUpdateSection(section, data);
      if (section === "database" && onDbSaved) {
        onDbSaved();
      }
      return result;
    },
  };

  function tabBadge(sections: string[]) {
    return sections.some((s) => restartRequired.includes(s)) ? (
      <span className="ml-1 inline-block h-1.5 w-1.5 rounded-full bg-yellow-500" />
    ) : null;
  }

  // Show non-tab step content (welcome, advanced summary, complete)
  const showNonTabStep = tour.active && tour.step.tab === null;

  // When closing the dialog during tour, finish the tour
  const handleOpenChange = (v: boolean) => {
    if (!v && tour.active) {
      tour.finish();
    }
    onOpenChange(v);
  };

  const TABS = [
    { value: "database", icon: Database, label: "Database", badge: undefined },
    { value: "retrieval", icon: Search, label: "Retrieval", badge: undefined },
    { value: "llm", icon: Brain, label: "LLM", badge: tabBadge(["llm", "cody", "ollama"]) },
    { value: "intelligence", icon: Sparkles, label: "Intel", badge: undefined },
    { value: "generation", icon: Sparkles, label: "Gen", badge: undefined },
    { value: "verification", icon: ShieldCheck, label: "Verify", badge: undefined },
    { value: "indexing", icon: Radio, label: "Index", badge: tabBadge(["indexing"]) },
  ];

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-h-[85vh] max-w-[620px] overflow-y-auto overflow-x-hidden">
        <DialogHeader>
          <DialogTitle>
            {tour.active ? "Setup Guide" : "Settings"}
          </DialogTitle>
          <DialogDescription>
            {tour.active
              ? "Follow the steps below to configure your system."
              : "Configure the Doc QA system. Safe settings apply immediately."}
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="text-muted-foreground h-6 w-6 animate-spin" />
          </div>
        ) : (
          <>
            {/* Tour overlay for non-tab steps */}
            {showNonTabStep && (
              <div className="mt-2">
                <TourOverlay tour={tour} />
              </div>
            )}

            {/* Tabs area — hidden when showing a non-tab tour step */}
            {!showNonTabStep && (
              <Tabs
                value={activeTab}
                onValueChange={(v) => {
                  if (!tour.active) setActiveTab(v);
                }}
                className="mt-2 overflow-hidden min-w-0"
              >
                <TabsList className="settings-tabs-list relative">
                  {TABS.map((tab) => (
                    <TabsTrigger
                      key={tab.value}
                      value={tab.value}
                      className="relative gap-1 text-xs"
                      disabled={tour.active}
                    >
                      <tab.icon className="h-3.5 w-3.5" /> {tab.label}{tab.badge}
                      {activeTab === tab.value && (
                        <motion.div
                          layoutId="activeSettingsTab"
                          className="absolute -bottom-px left-1 right-1 h-0.5 rounded-full bg-primary"
                          transition={{ type: "spring", stiffness: 400, damping: 28 }}
                        />
                      )}
                    </TabsTrigger>
                  ))}
                </TabsList>

                {/* Tour overlay for tab steps */}
                {tour.active && tour.step.tab && (
                  <div className="mt-3">
                    <TourOverlay tour={tour} />
                  </div>
                )}

                <TabsContent value="database" className="mt-4">
                  <DatabaseTab settings={wrappedSettings} />
                </TabsContent>
                <TabsContent value="retrieval" className="mt-4">
                  <RetrievalTab settings={wrappedSettings} />
                </TabsContent>
                <TabsContent value="llm" className="mt-4">
                  <LLMTab settings={wrappedSettings} />
                </TabsContent>
                <TabsContent value="intelligence" className="mt-4">
                  <IntelligenceTab settings={wrappedSettings} />
                </TabsContent>
                <TabsContent value="generation" className="mt-4">
                  <GenerationTab settings={wrappedSettings} />
                </TabsContent>
                <TabsContent value="verification" className="mt-4">
                  <VerificationTab settings={wrappedSettings} />
                </TabsContent>
                <TabsContent value="indexing" className="mt-4">
                  <IndexingTab settings={wrappedSettings} indexing={indexing} />
                </TabsContent>
              </Tabs>
            )}

            {/* Footer — "Take a Tour" button (only in settings, not duplicated) */}
            {!tour.active && (
              <div className="mt-4 flex justify-center border-t border-border/40 pt-3">
                <button
                  type="button"
                  onClick={() => tour.start()}
                  className="inline-flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
                >
                  <RotateCcw className="h-3 w-3" />
                  Take a Tour
                </button>
              </div>
            )}
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

import { useCallback, useEffect, useState } from "react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import type { UseSettingsReturn } from "@/hooks/use-settings";
import { field, SaveButton } from "./shared";

export function IntelligenceTab({ settings }: { settings: UseSettingsReturn }) {
    const { config, updateSection, saving } = settings;
    const [form, setForm] = useState({
        enable_intent_classification: true,
        intent_confidence_high: 0.85,
        intent_confidence_medium: 0.65,
        enable_multi_intent: true,
        max_sub_queries: 3,
    });
    const [saved, setSaved] = useState(false);

    useEffect(() => {
        if (!config) return;
        setForm({
            enable_intent_classification: field(config, "intelligence", "enable_intent_classification", true),
            intent_confidence_high: field(config, "intelligence", "intent_confidence_high", 0.85),
            intent_confidence_medium: field(config, "intelligence", "intent_confidence_medium", 0.65),
            enable_multi_intent: field(config, "intelligence", "enable_multi_intent", true),
            max_sub_queries: field(config, "intelligence", "max_sub_queries", 3),
        });
    }, [config]);

    const handleSave = useCallback(async () => {
        await updateSection("intelligence", form);
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    }, [form, updateSection]);

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-3">
                <Switch checked={form.enable_intent_classification} onCheckedChange={(v) => setForm((f) => ({ ...f, enable_intent_classification: v }))} id="intent-class" />
                <Label htmlFor="intent-class">Intent Classification</Label>
            </div>
            <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                    <Label htmlFor="conf-high">High Confidence</Label>
                    <Input id="conf-high" type="number" step={0.05} min={0} max={1} value={form.intent_confidence_high} onChange={(e) => setForm((f) => ({ ...f, intent_confidence_high: +e.target.value }))} />
                </div>
                <div className="space-y-2">
                    <Label htmlFor="conf-med">Medium Confidence</Label>
                    <Input id="conf-med" type="number" step={0.05} min={0} max={1} value={form.intent_confidence_medium} onChange={(e) => setForm((f) => ({ ...f, intent_confidence_medium: +e.target.value }))} />
                </div>
            </div>
            <div className="flex items-center gap-3">
                <Switch checked={form.enable_multi_intent} onCheckedChange={(v) => setForm((f) => ({ ...f, enable_multi_intent: v }))} id="multi-intent" />
                <Label htmlFor="multi-intent">Multi-Intent Decomposition</Label>
            </div>
            <div className="space-y-2">
                <Label htmlFor="max-sub">Max Sub-Queries</Label>
                <Input id="max-sub" type="number" min={1} max={10} value={form.max_sub_queries} onChange={(e) => setForm((f) => ({ ...f, max_sub_queries: +e.target.value }))} className="w-24" />
            </div>
            <SaveButton onClick={handleSave} saving={saving} saved={saved} />
        </div>
    );
}

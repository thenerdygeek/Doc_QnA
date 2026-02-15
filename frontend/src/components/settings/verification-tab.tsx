import { useCallback, useEffect, useState } from "react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import type { UseSettingsReturn } from "@/hooks/use-settings";
import { field, SaveButton } from "./shared";

export function VerificationTab({ settings }: { settings: UseSettingsReturn }) {
    const { config, updateSection, saving } = settings;
    const [form, setForm] = useState({
        enable_verification: true,
        enable_crag: true,
        confidence_threshold: 0.4,
        max_crag_rewrites: 2,
        abstain_on_low_confidence: true,
    });
    const [saved, setSaved] = useState(false);

    useEffect(() => {
        if (!config) return;
        setForm({
            enable_verification: field(config, "verification", "enable_verification", true),
            enable_crag: field(config, "verification", "enable_crag", true),
            confidence_threshold: field(config, "verification", "confidence_threshold", 0.4),
            max_crag_rewrites: field(config, "verification", "max_crag_rewrites", 2),
            abstain_on_low_confidence: field(config, "verification", "abstain_on_low_confidence", true),
        });
    }, [config]);

    const handleSave = useCallback(async () => {
        await updateSection("verification", form);
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    }, [form, updateSection]);

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-3">
                <Switch checked={form.enable_verification} onCheckedChange={(v) => setForm((f) => ({ ...f, enable_verification: v }))} id="verify" />
                <Label htmlFor="verify">Enable Verification</Label>
            </div>
            <div className="flex items-center gap-3">
                <Switch checked={form.enable_crag} onCheckedChange={(v) => setForm((f) => ({ ...f, enable_crag: v }))} id="crag" />
                <Label htmlFor="crag">Enable CRAG</Label>
            </div>
            <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                    <Label htmlFor="conf-thresh">Confidence Threshold</Label>
                    <Input id="conf-thresh" type="number" step={0.05} min={0} max={1} value={form.confidence_threshold} onChange={(e) => setForm((f) => ({ ...f, confidence_threshold: +e.target.value }))} />
                </div>
                <div className="space-y-2">
                    <Label htmlFor="max-crag">Max CRAG Rewrites</Label>
                    <Input id="max-crag" type="number" min={0} max={10} value={form.max_crag_rewrites} onChange={(e) => setForm((f) => ({ ...f, max_crag_rewrites: +e.target.value }))} />
                </div>
            </div>
            <div className="flex items-center gap-3">
                <Switch checked={form.abstain_on_low_confidence} onCheckedChange={(v) => setForm((f) => ({ ...f, abstain_on_low_confidence: v }))} id="abstain" />
                <Label htmlFor="abstain">Abstain on Low Confidence</Label>
            </div>
            <SaveButton onClick={handleSave} saving={saving} saved={saved} />
        </div>
    );
}

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
        confidence_threshold: 0.6,
        caveat_threshold: 0.4,
        max_crag_rewrites: 2,
        abstain_on_low_confidence: true,
    });
    const [feedbackForm, setFeedbackForm] = useState({
        enable_score_boost: false,
        boost_max: 0.10,
    });
    const [saved, setSaved] = useState(false);
    const [fbSaved, setFbSaved] = useState(false);

    useEffect(() => {
        if (!config) return;
        setForm({
            enable_verification: field(config, "verification", "enable_verification", true),
            enable_crag: field(config, "verification", "enable_crag", true),
            confidence_threshold: field(config, "verification", "confidence_threshold", 0.6),
            caveat_threshold: field(config, "verification", "caveat_threshold", 0.4),
            max_crag_rewrites: field(config, "verification", "max_crag_rewrites", 2),
            abstain_on_low_confidence: field(config, "verification", "abstain_on_low_confidence", true),
        });
        setFeedbackForm({
            enable_score_boost: field(config, "feedback", "enable_score_boost", false),
            boost_max: field(config, "feedback", "boost_max", 0.10),
        });
    }, [config]);

    const handleSave = useCallback(async () => {
        await updateSection("verification", form);
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    }, [form, updateSection]);

    const handleFbSave = useCallback(async () => {
        await updateSection("feedback", feedbackForm);
        setFbSaved(true);
        setTimeout(() => setFbSaved(false), 2000);
    }, [feedbackForm, updateSection]);

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-3">
                <Switch checked={form.enable_verification} onCheckedChange={(v) => setForm((f) => ({ ...f, enable_verification: v }))} id="verify" />
                <Label htmlFor="verify">Enable Verification</Label>
            </div>
            <div className="flex items-center gap-3">
                <Switch checked={form.enable_crag} onCheckedChange={(v) => setForm((f) => ({ ...f, enable_crag: v }))} id="crag" />
                <div>
                    <Label htmlFor="crag">Enable CRAG</Label>
                    <p className="text-[11px] text-muted-foreground">Rewrite queries when retrieval confidence is low</p>
                </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                    <Label htmlFor="conf-thresh">Confidence Threshold</Label>
                    <Input id="conf-thresh" type="number" step={0.05} min={0} max={1} value={form.confidence_threshold} onChange={(e) => setForm((f) => ({ ...f, confidence_threshold: +e.target.value }))} />
                </div>
                <div className="space-y-2">
                    <Label htmlFor="caveat-thresh">Caveat Threshold</Label>
                    <Input id="caveat-thresh" type="number" step={0.05} min={0} max={1} value={form.caveat_threshold} onChange={(e) => setForm((f) => ({ ...f, caveat_threshold: +e.target.value }))} />
                    <p className="text-[11px] text-muted-foreground">Below this, add a caveat to the answer</p>
                </div>
                <div className="space-y-2">
                    <Label htmlFor="max-crag">Max CRAG Rewrites</Label>
                    <Input id="max-crag" type="number" min={0} max={10} value={form.max_crag_rewrites} onChange={(e) => setForm((f) => ({ ...f, max_crag_rewrites: +e.target.value }))} />
                </div>
            </div>
            <div className="flex items-center gap-3">
                <Switch checked={form.abstain_on_low_confidence} onCheckedChange={(v) => setForm((f) => ({ ...f, abstain_on_low_confidence: v }))} id="abstain" />
                <div>
                    <Label htmlFor="abstain">Abstain on Low Confidence</Label>
                    <p className="text-[11px] text-muted-foreground">Refuse to answer when confidence is too low</p>
                </div>
            </div>
            <SaveButton onClick={handleSave} saving={saving} saved={saved} />

            {/* ── Feedback-based score boosting ─────────────────── */}
            <div className="border-t border-border/40 pt-4">
                <h4 className="mb-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Feedback Score Boost
                </h4>
                <div className="space-y-4">
                    <div className="flex items-center gap-3">
                        <Switch checked={feedbackForm.enable_score_boost} onCheckedChange={(v) => setFeedbackForm((f) => ({ ...f, enable_score_boost: v }))} id="score-boost" />
                        <div>
                            <Label htmlFor="score-boost">Enable Score Boosting</Label>
                            <p className="text-[11px] text-muted-foreground">Adjust retrieval scores based on user feedback history</p>
                        </div>
                    </div>
                    {feedbackForm.enable_score_boost && (
                        <div className="ml-10 space-y-2">
                            <Label htmlFor="boost-max">Max Boost</Label>
                            <Input id="boost-max" type="number" step={0.01} min={0} max={0.5} value={feedbackForm.boost_max} onChange={(e) => setFeedbackForm((f) => ({ ...f, boost_max: +e.target.value }))} className="w-24" />
                            <p className="text-[11px] text-muted-foreground">Maximum absolute score shift from feedback (0.10 = 10%)</p>
                        </div>
                    )}
                </div>
                <SaveButton onClick={handleFbSave} saving={saving} saved={fbSaved} />
            </div>
        </div>
    );
}

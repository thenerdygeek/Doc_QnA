import { useCallback, useEffect, useState } from "react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import type { UseSettingsReturn } from "@/hooks/use-settings";
import { field, SaveButton } from "./shared";

export function GenerationTab({ settings }: { settings: UseSettingsReturn }) {
    const { config, updateSection, saving } = settings;
    const [form, setForm] = useState({
        enable_diagrams: true,
        mermaid_validation: "auto",
        max_diagram_retries: 3,
    });
    const [saved, setSaved] = useState(false);

    useEffect(() => {
        if (!config) return;
        setForm({
            enable_diagrams: field(config, "generation", "enable_diagrams", true),
            mermaid_validation: field(config, "generation", "mermaid_validation", "auto"),
            max_diagram_retries: field(config, "generation", "max_diagram_retries", 3),
        });
    }, [config]);

    const handleSave = useCallback(async () => {
        await updateSection("generation", form);
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    }, [form, updateSection]);

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-3">
                <Switch checked={form.enable_diagrams} onCheckedChange={(v) => setForm((f) => ({ ...f, enable_diagrams: v }))} id="diagrams" />
                <Label htmlFor="diagrams">Enable Diagrams</Label>
            </div>
            <div className="space-y-2">
                <Label>Mermaid Validation</Label>
                <Select value={form.mermaid_validation} onValueChange={(v) => setForm((f) => ({ ...f, mermaid_validation: v }))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                        <SelectItem value="auto">Auto</SelectItem>
                        <SelectItem value="node">Node</SelectItem>
                        <SelectItem value="regex">Regex</SelectItem>
                        <SelectItem value="none">None</SelectItem>
                    </SelectContent>
                </Select>
            </div>
            <div className="space-y-2">
                <Label htmlFor="max-retries">Max Diagram Retries</Label>
                <Input id="max-retries" type="number" min={0} max={10} value={form.max_diagram_retries} onChange={(e) => setForm((f) => ({ ...f, max_diagram_retries: +e.target.value }))} className="w-24" />
            </div>
            <SaveButton onClick={handleSave} saving={saving} saved={saved} />
        </div>
    );
}

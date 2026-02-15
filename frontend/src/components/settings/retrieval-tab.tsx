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

export function RetrievalTab({ settings }: { settings: UseSettingsReturn }) {
    const { config, updateSection, saving } = settings;
    const [form, setForm] = useState({
        search_mode: "hybrid",
        top_k: 5,
        candidate_pool: 20,
        min_score: 0.3,
        max_chunks_per_file: 2,
        rerank: true,
    });
    const [saved, setSaved] = useState(false);

    useEffect(() => {
        if (!config) return;
        setForm({
            search_mode: field(config, "retrieval", "search_mode", "hybrid"),
            top_k: field(config, "retrieval", "top_k", 5),
            candidate_pool: field(config, "retrieval", "candidate_pool", 20),
            min_score: field(config, "retrieval", "min_score", 0.3),
            max_chunks_per_file: field(config, "retrieval", "max_chunks_per_file", 2),
            rerank: field(config, "retrieval", "rerank", true),
        });
    }, [config]);

    const handleSave = useCallback(async () => {
        await updateSection("retrieval", form);
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    }, [form, updateSection]);

    return (
        <div className="space-y-4">
            <div className="space-y-2">
                <Label>Search Mode</Label>
                <Select value={form.search_mode} onValueChange={(v) => setForm((f) => ({ ...f, search_mode: v }))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                        <SelectItem value="hybrid">Hybrid</SelectItem>
                        <SelectItem value="vector">Vector</SelectItem>
                        <SelectItem value="fts">Full-Text Search</SelectItem>
                    </SelectContent>
                </Select>
            </div>
            <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                    <Label htmlFor="top-k">Top K</Label>
                    <Input id="top-k" type="number" min={1} max={50} value={form.top_k} onChange={(e) => setForm((f) => ({ ...f, top_k: +e.target.value }))} />
                </div>
                <div className="space-y-2">
                    <Label htmlFor="candidate-pool">Candidate Pool</Label>
                    <Input id="candidate-pool" type="number" min={1} max={200} value={form.candidate_pool} onChange={(e) => setForm((f) => ({ ...f, candidate_pool: +e.target.value }))} />
                </div>
                <div className="space-y-2">
                    <Label htmlFor="min-score">Min Score</Label>
                    <Input id="min-score" type="number" step={0.05} min={0} max={1} value={form.min_score} onChange={(e) => setForm((f) => ({ ...f, min_score: +e.target.value }))} />
                </div>
                <div className="space-y-2">
                    <Label htmlFor="max-chunks">Max Chunks/File</Label>
                    <Input id="max-chunks" type="number" min={1} max={20} value={form.max_chunks_per_file} onChange={(e) => setForm((f) => ({ ...f, max_chunks_per_file: +e.target.value }))} />
                </div>
            </div>
            <div className="flex items-center gap-3">
                <Switch checked={form.rerank} onCheckedChange={(v) => setForm((f) => ({ ...f, rerank: v }))} id="rerank" />
                <Label htmlFor="rerank">Enable Reranking</Label>
            </div>
            <SaveButton onClick={handleSave} saving={saving} saved={saved} />
        </div>
    );
}

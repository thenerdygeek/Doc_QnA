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
        enable_query_expansion: false,
        max_expansion_queries: 3,
        enable_hyde: false,
        enable_query_rewriting: true,
        enable_multi_hop: false,
        max_hops: 2,
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
            enable_query_expansion: field(config, "retrieval", "enable_query_expansion", false),
            max_expansion_queries: field(config, "retrieval", "max_expansion_queries", 3),
            enable_hyde: field(config, "retrieval", "enable_hyde", false),
            enable_query_rewriting: field(config, "retrieval", "enable_query_rewriting", true),
            enable_multi_hop: field(config, "multi_hop", "enable_multi_hop", false),
            max_hops: field(config, "multi_hop", "max_hops", 2),
        });
    }, [config]);

    const handleSave = useCallback(async () => {
        // Split retrieval vs multi_hop fields
        const { enable_multi_hop, max_hops, ...retrievalFields } = form;
        await updateSection("retrieval", retrievalFields);
        await updateSection("multi_hop", { enable_multi_hop, max_hops });
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

            {/* ── Advanced retrieval ──────────────────────────── */}
            <div className="border-t border-border/40 pt-4">
                <h4 className="mb-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Advanced Retrieval
                </h4>
                <div className="space-y-4">
                    <div className="space-y-3">
                        <div className="flex items-center gap-3">
                            <Switch checked={form.enable_query_expansion} onCheckedChange={(v) => setForm((f) => ({ ...f, enable_query_expansion: v }))} id="query-expansion" />
                            <div>
                                <Label htmlFor="query-expansion">Query Expansion</Label>
                                <p className="text-[11px] text-muted-foreground">Generate alternative phrasings to improve recall</p>
                            </div>
                        </div>
                        {form.enable_query_expansion && (
                            <div className="ml-10 space-y-2">
                                <Label htmlFor="max-queries">Max Expansion Queries</Label>
                                <Input id="max-queries" type="number" min={1} max={10} value={form.max_expansion_queries} onChange={(e) => setForm((f) => ({ ...f, max_expansion_queries: +e.target.value }))} className="w-24" />
                            </div>
                        )}
                    </div>
                    <div className="flex items-center gap-3">
                        <Switch checked={form.enable_hyde} onCheckedChange={(v) => setForm((f) => ({ ...f, enable_hyde: v }))} id="hyde" />
                        <div>
                            <Label htmlFor="hyde">HyDE</Label>
                            <p className="text-[11px] text-muted-foreground">Generate a hypothetical answer to improve embedding search</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        <Switch checked={form.enable_query_rewriting} onCheckedChange={(v) => setForm((f) => ({ ...f, enable_query_rewriting: v }))} id="query-rewriting" />
                        <div>
                            <Label htmlFor="query-rewriting">Query Rewriting</Label>
                            <p className="text-[11px] text-muted-foreground">Rewrite follow-up questions into standalone queries for better retrieval</p>
                        </div>
                    </div>
                    <div className="space-y-3">
                        <div className="flex items-center gap-3">
                            <Switch checked={form.enable_multi_hop} onCheckedChange={(v) => setForm((f) => ({ ...f, enable_multi_hop: v }))} id="multi-hop" />
                            <div>
                                <Label htmlFor="multi-hop">Multi-hop Reasoning</Label>
                                <p className="text-[11px] text-muted-foreground">Detect knowledge gaps and retrieve additional context iteratively</p>
                            </div>
                        </div>
                        {form.enable_multi_hop && (
                            <div className="ml-10 space-y-2">
                                <Label htmlFor="max-hops">Max Hops</Label>
                                <Input id="max-hops" type="number" min={1} max={5} value={form.max_hops} onChange={(e) => setForm((f) => ({ ...f, max_hops: +e.target.value }))} className="w-24" />
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <SaveButton onClick={handleSave} saving={saving} saved={saved} />
        </div>
    );
}

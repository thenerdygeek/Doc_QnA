import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
    Loader2,
    CheckCircle,
    XCircle,
    AlertTriangle,
    FolderOpen,
    FolderSearch,
    Play,
    Square,
    FileText,
    Folder,
    ArrowUp,
} from "lucide-react";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { api } from "@/api/client";
import type { UseSettingsReturn } from "@/hooks/use-settings";
import type { UseIndexingReturn } from "@/hooks/use-indexing";
import { field, RestartBadge, SaveButton } from "./shared";

// ── State label helpers ──────────────────────────────────────────

const STATE_LABELS: Record<string, string> = {
    scanning: "Scanning files\u2026",
    indexing: "Indexing documents\u2026",
    rebuilding_fts: "Building search index\u2026",
    swapping: "Finalizing\u2026",
};

function stateLabel(state: string): string {
    return STATE_LABELS[state] ?? state;
}

/**
 * Extract a short display path from a full file path.
 *
 * Shows up to the last `maxSegments` path segments, prefixed with "\u2026/"
 * when truncated.  Handles both `/` (Unix) and `\` (Windows) separators.
 */
function shortenPath(fullPath: string, repoRoot: string, maxSegments = 3): string {
    // Normalise separators to /
    const norm = fullPath.replace(/\\/g, "/");
    const normRoot = repoRoot.replace(/\\/g, "/").replace(/\/+$/, "");

    // Strip the repo root prefix to get a relative path
    let rel = norm;
    if (normRoot && norm.startsWith(normRoot)) {
        rel = norm.slice(normRoot.length).replace(/^\//, "");
    }

    const parts = rel.split("/").filter(Boolean);
    if (parts.length <= maxSegments) return parts.join("/");
    return "\u2026/" + parts.slice(-maxSegments).join("/");
}

// ── Folder browser dialog ──────────────────────────────────────

function FolderBrowser({ open, onClose, onSelect }: { open: boolean; onClose: () => void; onSelect: (path: string) => void }) {
    const [currentPath, setCurrentPath] = useState("");
    const [parentPath, setParentPath] = useState("");
    const [dirs, setDirs] = useState<{ name: string; path: string }[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const browse = useCallback(async (path: string) => {
        setLoading(true);
        setError("");
        try {
            const result = await api.browse(path);
            setCurrentPath(result.path);
            setParentPath(result.parent);
            setDirs(result.dirs);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to browse");
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (open) browse("");
    }, [open, browse]);

    if (!open) return null;

    return (
        <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
            <DialogContent className="max-w-md">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2"><FolderSearch className="h-4 w-4" /> Browse Folders</DialogTitle>
                    <DialogDescription>Navigate to a documentation folder and click Select.</DialogDescription>
                </DialogHeader>

                {/* Current path */}
                <div className="rounded-md bg-muted/50 px-3 py-2">
                    <code className="block truncate text-xs text-foreground">{currentPath || "/"}</code>
                </div>

                {/* Directory listing */}
                <ScrollArea className="h-[280px] rounded-md border">
                    <div className="p-1">
                        {/* Up button */}
                        {parentPath && (
                            <button
                                onClick={() => browse(parentPath)}
                                className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent"
                            >
                                <ArrowUp className="h-3.5 w-3.5 text-muted-foreground" />
                                <span className="text-muted-foreground">..</span>
                            </button>
                        )}
                        {loading ? (
                            <div className="flex items-center justify-center py-8">
                                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                            </div>
                        ) : error ? (
                            <p className="px-2 py-4 text-center text-xs text-red-500">{error}</p>
                        ) : dirs.length === 0 ? (
                            <p className="px-2 py-4 text-center text-xs text-muted-foreground">No subdirectories</p>
                        ) : (
                            dirs.map((d) => (
                                <button
                                    key={d.path}
                                    onClick={() => browse(d.path)}
                                    className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent"
                                >
                                    <Folder className="h-3.5 w-3.5 text-blue-500" />
                                    <span className="truncate">{d.name}</span>
                                </button>
                            ))
                        )}
                    </div>
                </ScrollArea>

                {/* Actions */}
                <div className="flex justify-end gap-2">
                    <Button variant="outline" size="sm" onClick={onClose}>Cancel</Button>
                    <Button size="sm" disabled={!currentPath} onClick={() => { onSelect(currentPath); onClose(); }}>
                        <FolderOpen className="mr-1.5 h-3.5 w-3.5" /> Select
                    </Button>
                </div>
            </DialogContent>
        </Dialog>
    );
}

// ── Indexing Tab ─────────────────────────────────────────────────

export function IndexingTab({ settings, indexing }: { settings: UseSettingsReturn; indexing: UseIndexingReturn }) {
    const { config, updateSection, saving, restartRequired } = settings;
    const [repoPath, setRepoPath] = useState("");
    const [browseOpen, setBrowseOpen] = useState(false);
    const [form, setForm] = useState({
        chunk_size: 512,
        chunk_overlap: 50,
        min_chunk_size: 100,
        embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
    });
    const [saved, setSaved] = useState(false);

    useEffect(() => {
        if (!config) return;
        setRepoPath(field(config, "doc_repo", "path", ""));
        setForm({
            chunk_size: field(config, "indexing", "chunk_size", 512),
            chunk_overlap: field(config, "indexing", "chunk_overlap", 50),
            min_chunk_size: field(config, "indexing", "min_chunk_size", 100),
            embedding_model: field(config, "indexing", "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        });
    }, [config]);

    // Auto-reconnect to running job when tab mounts
    useEffect(() => {
        if (indexing.phase === "idle") {
            indexing.reconnect();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const needsRestart = restartRequired.includes("indexing");
    const isRunning = indexing.phase === "running";

    const handleStartIndexing = useCallback(() => {
        if (!repoPath.trim()) return;
        indexing.start(repoPath.trim());
    }, [repoPath, indexing]);

    const handleSave = useCallback(async () => {
        await updateSection("indexing", form);
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    }, [form, updateSection]);

    return (
        <div className="space-y-5 overflow-hidden">
            {/* ── Repo path + Start/Cancel ─────────────────────────── */}
            <div className="space-y-2">
                <Label htmlFor="repo-path">Documentation Repository Path</Label>
                <div className="flex gap-2">
                    <div className="relative min-w-0 flex-1">
                        <FolderOpen className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                        <Input
                            id="repo-path"
                            placeholder="/path/to/docs"
                            value={repoPath}
                            onChange={(e) => setRepoPath(e.target.value)}
                            disabled={isRunning}
                            className="pl-8"
                        />
                    </div>
                    <Button
                        size="sm"
                        variant="outline"
                        onClick={() => setBrowseOpen(true)}
                        disabled={isRunning}
                        className="shrink-0"
                        title="Browse folders"
                    >
                        <FolderSearch className="h-4 w-4" />
                    </Button>
                    {isRunning ? (
                        <Button
                            size="sm"
                            variant="destructive"
                            onClick={() => indexing.cancel()}
                            className="shrink-0"
                        >
                            <Square className="mr-1 h-3 w-3" /> Cancel
                        </Button>
                    ) : (
                        <Button
                            size="sm"
                            onClick={handleStartIndexing}
                            disabled={!repoPath.trim()}
                            className="shrink-0"
                        >
                            <Play className="mr-1 h-3 w-3" /> Start Indexing
                        </Button>
                    )}
                </div>
                <p className="text-muted-foreground text-xs">
                    Path to a documentation folder (relative or absolute). A new index will replace the old one.
                </p>
                <FolderBrowser open={browseOpen} onClose={() => setBrowseOpen(false)} onSelect={setRepoPath} />
            </div>

            {/* ── Progress section ──────────────────────────────────── */}
            {isRunning && (
                <div className="w-full min-w-0 space-y-3 overflow-hidden rounded-lg border border-blue-500/30 bg-blue-500/5 p-3 dark:bg-blue-500/10">
                    <div className="flex items-center justify-between">
                        <span className="flex items-center gap-1.5 text-sm font-medium text-blue-700 dark:text-blue-300">
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            {stateLabel(indexing.state)}
                        </span>
                        <span className="text-xs tabular-nums text-muted-foreground">
                            {indexing.processedFiles} / {indexing.totalFiles} files
                        </span>
                    </div>

                    {/* Animated progress bar */}
                    <div className="h-2 overflow-hidden rounded-full bg-blue-500/10">
                        <motion.div
                            className="h-full rounded-full bg-blue-500"
                            initial={{ width: 0 }}
                            animate={{ width: `${indexing.percent}%` }}
                            transition={{ ease: "easeOut", duration: 0.3 }}
                        />
                    </div>

                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>{indexing.percent}%</span>
                        <span>{indexing.totalChunks} chunks</span>
                    </div>

                    {/* Recent files log */}
                    {indexing.recentFiles.length > 0 && (
                        <ScrollArea className="h-24 overflow-x-hidden rounded border border-border/50 bg-background/50 p-2">
                            <div className="space-y-0.5 overflow-hidden">
                                {indexing.recentFiles.map((f, i) => (
                                    <div
                                        key={`${f.file}-${i}`}
                                        className="flex min-w-0 items-center gap-1.5 overflow-hidden text-[11px] text-muted-foreground"
                                    >
                                        <FileText className="h-3 w-3 shrink-0" />
                                        <span className="min-w-0 truncate" title={f.file}>
                                            {shortenPath(f.file, indexing.repoPath)}
                                        </span>
                                        {f.skipped ? (
                                            <span className="ml-auto shrink-0 text-yellow-600 dark:text-yellow-400">skipped</span>
                                        ) : (
                                            <span className="ml-auto shrink-0 tabular-nums">{f.chunks} chunks</span>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </ScrollArea>
                    )}
                </div>
            )}

            {/* ── Done section ──────────────────────────────────────── */}
            {indexing.phase === "done" && (
                <div className="flex items-start gap-2 rounded-lg border border-green-500/30 bg-green-500/5 p-3 dark:bg-green-500/10">
                    <CheckCircle className="mt-0.5 h-4 w-4 shrink-0 text-green-600 dark:text-green-400" />
                    <div>
                        <p className="text-sm font-medium text-green-700 dark:text-green-300">
                            Indexing complete
                        </p>
                        <p className="mt-0.5 text-xs text-muted-foreground">
                            {indexing.totalFiles} files, {indexing.totalChunks} chunks in{" "}
                            {indexing.elapsed !== null ? `${indexing.elapsed}s` : "—"}
                        </p>
                    </div>
                </div>
            )}

            {/* ── Cancelled section ─────────────────────────────────── */}
            {indexing.phase === "cancelled" && (
                <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 dark:bg-amber-500/10">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" />
                    <div>
                        <p className="text-sm font-medium text-amber-700 dark:text-amber-300">
                            Indexing cancelled
                        </p>
                        <p className="mt-0.5 text-xs text-muted-foreground">
                            Previous index preserved. You can start indexing again at any time.
                        </p>
                    </div>
                </div>
            )}

            {/* ── Error section ─────────────────────────────────────── */}
            {indexing.phase === "error" && (
                <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/5 p-3 dark:bg-red-500/10">
                    <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-600 dark:text-red-400" />
                    <div className="min-w-0">
                        <p className="text-sm font-medium text-red-700 dark:text-red-300">
                            Indexing failed
                        </p>
                        <p className="mt-0.5 break-all text-xs text-muted-foreground">
                            {indexing.error}
                        </p>
                    </div>
                </div>
            )}

            {/* ── Chunk settings ────────────────────────────────────── */}
            <div className="border-t border-border/40 pt-4">
                <h4 className="mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    Chunk Settings
                </h4>
                {needsRestart && <RestartBadge />}
                <div className="mt-2 grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                        <Label htmlFor="chunk-size">Chunk Size</Label>
                        <Input id="chunk-size" type="number" min={64} max={4096} value={form.chunk_size} onChange={(e) => setForm((f) => ({ ...f, chunk_size: +e.target.value }))} />
                    </div>
                    <div className="space-y-2">
                        <Label htmlFor="chunk-overlap">Chunk Overlap</Label>
                        <Input id="chunk-overlap" type="number" min={0} max={512} value={form.chunk_overlap} onChange={(e) => setForm((f) => ({ ...f, chunk_overlap: +e.target.value }))} />
                    </div>
                    <div className="space-y-2">
                        <Label htmlFor="min-chunk">Min Chunk Size</Label>
                        <Input id="min-chunk" type="number" min={10} max={1024} value={form.min_chunk_size} onChange={(e) => setForm((f) => ({ ...f, min_chunk_size: +e.target.value }))} />
                    </div>
                </div>
                <div className="mt-4 space-y-2">
                    <Label htmlFor="embed-model">Embedding Model</Label>
                    <Input id="embed-model" value={form.embedding_model} onChange={(e) => setForm((f) => ({ ...f, embedding_model: e.target.value }))} />
                </div>
                <SaveButton onClick={handleSave} saving={saving} saved={saved} />
            </div>
        </div>
    );
}

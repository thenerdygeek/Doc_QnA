import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Brain, HardDrive, XCircle } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import type { CodyTestResponse, OllamaTestResponse } from "@/types/api";
import { api } from "@/api/client";
import type { UseSettingsReturn } from "@/hooks/use-settings";
import { field, RestartBadge, SaveButton, AnimatedCheckmark, LoadingDots } from "./shared";

export function LLMTab({ settings }: { settings: UseSettingsReturn }) {
    const { config, updateSection, saving, restartRequired } = settings;
    const [llm, setLlm] = useState({ primary: "cody", fallback: "ollama" });
    const [cody, setCody] = useState({ model: "", endpoint: "", access_token_env: "SRC_ACCESS_TOKEN", _token_is_set: false, _token_masked: "" });
    const [ollama, setOllama] = useState({ host: "", model: "" });
    const [saved, setSaved] = useState(false);

    // Test connection state
    const [codyTest, setCodyTest] = useState<{ loading: boolean; result: CodyTestResponse | null }>({ loading: false, result: null });
    const [ollamaTest, setOllamaTest] = useState<{ loading: boolean; result: OllamaTestResponse | null }>({ loading: false, result: null });

    useEffect(() => {
        if (!config) return;
        setLlm({
            primary: field(config, "llm", "primary", "cody"),
            fallback: field(config, "llm", "fallback", "ollama"),
        });
        setCody({
            model: field(config, "cody", "model", ""),
            endpoint: field(config, "cody", "endpoint", ""),
            access_token_env: field(config, "cody", "access_token_env", "SRC_ACCESS_TOKEN"),
            _token_is_set: field(config, "cody", "_token_is_set", false),
            _token_masked: field(config, "cody", "_token_masked", ""),
        });
        setOllama({
            host: field(config, "ollama", "host", ""),
            model: field(config, "ollama", "model", ""),
        });
    }, [config]);

    const needsRestart = restartRequired.some((s) => ["llm", "cody", "ollama"].includes(s));

    const handleTestCody = useCallback(async () => {
        setCodyTest({ loading: true, result: null });
        try {
            const result = await api.llm.testCody({
                endpoint: cody.endpoint || "https://sourcegraph.com",
                access_token_env: cody.access_token_env || "SRC_ACCESS_TOKEN",
            });
            setCodyTest({ loading: false, result });
            // Auto-select current model if it exists in the list
            if (result.ok && result.models && result.models.length > 0) {
                const currentInList = result.models.find((m) => m.id === cody.model);
                if (!currentInList) {
                    setCody((f) => ({ ...f, model: result.models![0]!.id }));
                }
            }
        } catch (err) {
            setCodyTest({ loading: false, result: { ok: false, error: err instanceof Error ? err.message : "Connection failed" } });
        }
    }, [cody.endpoint, cody.access_token_env, cody.model]);

    const handleTestOllama = useCallback(async () => {
        setOllamaTest({ loading: true, result: null });
        try {
            const result = await api.llm.testOllama({
                host: ollama.host || "http://localhost:11434",
            });
            setOllamaTest({ loading: false, result });
            // Auto-select current model if it exists in the list
            if (result.ok && result.models && result.models.length > 0) {
                const currentInList = result.models.find((m) => m.id === ollama.model);
                if (!currentInList) {
                    setOllama((f) => ({ ...f, model: result.models![0]!.id }));
                }
            }
        } catch (err) {
            setOllamaTest({ loading: false, result: { ok: false, error: err instanceof Error ? err.message : "Connection failed" } });
        }
    }, [ollama.host, ollama.model]);

    const handleSave = useCallback(async () => {
        await updateSection("llm", llm);
        await updateSection("cody", { model: cody.model, endpoint: cody.endpoint, access_token_env: cody.access_token_env });
        await updateSection("ollama", ollama);
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    }, [llm, cody, ollama, updateSection]);

    return (
        <div className="space-y-4">
            {needsRestart && <RestartBadge />}
            <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                    <Label>Primary LLM</Label>
                    <Select value={llm.primary} onValueChange={(v) => setLlm((f) => ({ ...f, primary: v }))}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                            <SelectItem value="cody">Cody</SelectItem>
                            <SelectItem value="ollama">Ollama</SelectItem>
                        </SelectContent>
                    </Select>
                </div>
                <div className="space-y-2">
                    <Label>Fallback LLM</Label>
                    <Select value={llm.fallback} onValueChange={(v) => setLlm((f) => ({ ...f, fallback: v }))}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                            <SelectItem value="cody">Cody</SelectItem>
                            <SelectItem value="ollama">Ollama</SelectItem>
                        </SelectContent>
                    </Select>
                </div>
            </div>

            {/* ── Cody fieldset ──────────────────────────────────────── */}
            <fieldset className="space-y-3 rounded-md border p-3">
                <legend className="text-muted-foreground px-1 text-xs font-medium">Cody</legend>
                <div className="space-y-2">
                    <Label htmlFor="cody-endpoint">Endpoint</Label>
                    <Input id="cody-endpoint" value={cody.endpoint} placeholder="https://sourcegraph.com" onChange={(e) => setCody((f) => ({ ...f, endpoint: e.target.value }))} />
                </div>
                <div className="space-y-2">
                    <Label htmlFor="cody-token-env">Access Token Env</Label>
                    <Input id="cody-token-env" value={cody.access_token_env} placeholder="SRC_ACCESS_TOKEN" onChange={(e) => setCody((f) => ({ ...f, access_token_env: e.target.value }))} />
                    <p className="text-muted-foreground text-[11px]">
                        Environment variable containing your Sourcegraph token
                    </p>
                    {cody._token_is_set ? (
                        <div className="flex items-center gap-2 rounded-md border border-green-500/20 bg-green-500/5 px-2.5 py-1.5">
                            <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-500" />
                            <code className="flex-1 truncate font-mono text-xs text-muted-foreground">{cody._token_masked}</code>
                            <span className="text-[11px] font-medium text-green-600 dark:text-green-400">Active</span>
                        </div>
                    ) : (
                        <div className="flex items-center gap-2 rounded-md border border-red-500/20 bg-red-500/5 px-2.5 py-1.5">
                            <span className="inline-block h-1.5 w-1.5 rounded-full bg-red-500" />
                            <span className="flex-1 text-xs text-muted-foreground">No token found in <code className="font-mono">${cody.access_token_env}</code></span>
                            <span className="text-[11px] font-medium text-red-500 dark:text-red-400">Not set</span>
                        </div>
                    )}
                </div>

                {/* Test Connection */}
                <div className="flex items-center gap-2">
                    <Button size="sm" variant="outline" onClick={handleTestCody} disabled={codyTest.loading}>
                        {codyTest.loading ? <LoadingDots /> : <Brain className="mr-1 h-3 w-3" />}
                        {!codyTest.loading && "Test Connection"}
                    </Button>
                    {codyTest.result && !codyTest.result.ok && (
                        <motion.span initial={{ opacity: 0, x: -4 }} animate={{ opacity: 1, x: 0 }} className="inline-flex items-center gap-1 text-xs text-red-600 dark:text-red-400">
                            <XCircle className="h-3.5 w-3.5" /> {codyTest.result.error}
                        </motion.span>
                    )}
                </div>

                {/* Auth success display */}
                {codyTest.result?.ok && codyTest.result.user && (
                    <motion.div initial={{ opacity: 0, x: -4 }} animate={{ opacity: 1, x: 0 }} className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400">
                        <AnimatedCheckmark />
                        <span>
                            Authenticated as <strong>{codyTest.result.user.displayName}</strong>
                            {codyTest.result.user.email && (
                                <span className="text-muted-foreground"> ({codyTest.result.user.email})</span>
                            )}
                        </span>
                    </motion.div>
                )}

                {/* Model dropdown — shown after successful test */}
                {codyTest.result?.ok && codyTest.result.models && codyTest.result.models.length > 0 ? (
                    <div className="space-y-2">
                        <Label>Model</Label>
                        <Select value={cody.model} onValueChange={(v) => setCody((f) => ({ ...f, model: v }))}>
                            <SelectTrigger><SelectValue placeholder="Select a model" /></SelectTrigger>
                            <SelectContent>
                                {codyTest.result.models.map((m) => (
                                    <SelectItem key={m.id} value={m.id}>
                                        <span className="flex items-center gap-1.5">
                                            {m.thinking && <Brain className="h-3 w-3 text-purple-500" />}
                                            <span>{m.displayName}</span>
                                            <span className="text-muted-foreground text-[10px]">{m.provider}</span>
                                        </span>
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                ) : (
                    <div className="space-y-2">
                        <Label htmlFor="cody-model">Model</Label>
                        <Input id="cody-model" value={cody.model} onChange={(e) => setCody((f) => ({ ...f, model: e.target.value }))} placeholder="anthropic::2025-01-01::claude-3.5-sonnet" />
                        <p className="text-muted-foreground text-[11px]">
                            Test connection to discover available models
                        </p>
                    </div>
                )}
            </fieldset>

            {/* ── Ollama fieldset ────────────────────────────────────── */}
            <fieldset className="space-y-3 rounded-md border p-3">
                <legend className="text-muted-foreground px-1 text-xs font-medium">Ollama</legend>
                <div className="space-y-2">
                    <Label htmlFor="ollama-host">Host</Label>
                    <Input id="ollama-host" value={ollama.host} placeholder="http://localhost:11434" onChange={(e) => setOllama((f) => ({ ...f, host: e.target.value }))} />
                </div>

                {/* Test Connection */}
                <div className="flex items-center gap-2">
                    <Button size="sm" variant="outline" onClick={handleTestOllama} disabled={ollamaTest.loading}>
                        {ollamaTest.loading ? <LoadingDots /> : <HardDrive className="mr-1 h-3 w-3" />}
                        {!ollamaTest.loading && "Test Connection"}
                    </Button>
                    {ollamaTest.result && !ollamaTest.result.ok && (
                        <motion.span initial={{ opacity: 0, x: -4 }} animate={{ opacity: 1, x: 0 }} className="inline-flex items-center gap-1 text-xs text-red-600 dark:text-red-400">
                            <XCircle className="h-3.5 w-3.5" /> {ollamaTest.result.error}
                        </motion.span>
                    )}
                </div>

                {/* Success display */}
                {ollamaTest.result?.ok && (
                    <motion.div initial={{ opacity: 0, x: -4 }} animate={{ opacity: 1, x: 0 }} className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400">
                        <AnimatedCheckmark />
                        <span>Connected &mdash; {ollamaTest.result.models?.length ?? 0} models available</span>
                    </motion.div>
                )}

                {/* Model dropdown — shown after successful test */}
                {ollamaTest.result?.ok && ollamaTest.result.models && ollamaTest.result.models.length > 0 ? (
                    <div className="space-y-2">
                        <Label>Model</Label>
                        <Select value={ollama.model} onValueChange={(v) => setOllama((f) => ({ ...f, model: v }))}>
                            <SelectTrigger><SelectValue placeholder="Select a model" /></SelectTrigger>
                            <SelectContent>
                                {ollamaTest.result.models.map((m) => (
                                    <SelectItem key={m.id} value={m.id}>
                                        <span className="flex items-center gap-1.5">
                                            <span>{m.displayName}</span>
                                            {m.size && (
                                                <Badge variant="secondary" className="h-4 px-1 text-[9px]">{m.size}</Badge>
                                            )}
                                        </span>
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                ) : (
                    <div className="space-y-2">
                        <Label htmlFor="ollama-model">Model</Label>
                        <Input id="ollama-model" value={ollama.model} onChange={(e) => setOllama((f) => ({ ...f, model: e.target.value }))} placeholder="qwen2.5:7b" />
                        <p className="text-muted-foreground text-[11px]">
                            Test connection to discover available models
                        </p>
                    </div>
                )}
            </fieldset>

            <SaveButton onClick={handleSave} saving={saving} saved={saved} />
        </div>
    );
}

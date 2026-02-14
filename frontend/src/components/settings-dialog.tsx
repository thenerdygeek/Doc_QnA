import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  CheckCircle,
  XCircle,
  Loader2,
  AlertTriangle,
  Database,
  Search,
  Brain,
  Sparkles,
  ShieldCheck,
  Radio,
  HardDrive,
  ChevronRight,
  ChevronLeft,
  SkipForward,
  RotateCcw,
  PartyPopper,
  Rocket,
  FolderOpen,
  Play,
  Square,
  FileText,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { ConfigData, CodyTestResponse, OllamaTestResponse } from "@/types/api";
import { api } from "@/api/client";
import type { UseSettingsReturn } from "@/hooks/use-settings";
import type { UseTourReturn } from "@/hooks/use-tour";
import type { UseIndexingReturn } from "@/hooks/use-indexing";

// ── Helpers ─────────────────────────────────────────────────────

function field<T>(config: ConfigData | null, section: string, key: string, fallback: T): T {
  if (!config || !config[section]) return fallback;
  const v = config[section][key];
  return (v ?? fallback) as T;
}

function RestartBadge() {
  return (
    <Badge variant="outline" className="ml-2 border-yellow-500/50 text-yellow-600 text-[10px] dark:text-yellow-400">
      <AlertTriangle className="mr-1 h-3 w-3" /> restart required
    </Badge>
  );
}

function SaveButton({ onClick, saving, saved }: { onClick: () => void; saving: boolean; saved: boolean }) {
  return (
    <Button size="sm" onClick={onClick} disabled={saving} className="mt-4">
      {saving && <Loader2 className="mr-1 h-3 w-3 animate-spin" />}
      {saved ? "Saved" : "Save"}
    </Button>
  );
}

// ── Tour Overlay ────────────────────────────────────────────────

function TourOverlay({ tour }: { tour: UseTourReturn }) {
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

// ── Tab Components ──────────────────────────────────────────────

function DatabaseTab({ settings }: { settings: UseSettingsReturn }) {
  const { config, updateSection, saving, testDbConnection, dbTestResult, runMigrations, migrateResult } = settings;
  const [url, setUrl] = useState("");
  const [testing, setTesting] = useState(false);
  const [migrating, setMigrating] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setUrl(field(config, "database", "url", "") ?? "");
  }, [config]);

  const handleTest = useCallback(async () => {
    if (!url) return;
    setTesting(true);
    try { await testDbConnection(url); }
    finally { setTesting(false); }
  }, [url, testDbConnection]);

  const handleMigrate = useCallback(async () => {
    setMigrating(true);
    try { await runMigrations(); }
    finally { setMigrating(false); }
  }, [runMigrations]);

  const handleSave = useCallback(async () => {
    await updateSection("database", { url: url || null });
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }, [url, updateSection]);

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="db-url">Database URL</Label>
        <Input
          id="db-url"
          placeholder="postgresql://user:pass@localhost:5432/doc_qa"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
        <p className="text-muted-foreground text-xs">
          PostgreSQL connection string for conversation persistence.
        </p>
      </div>

      <div className="flex items-center gap-2">
        <Button size="sm" variant="outline" onClick={handleTest} disabled={testing || !url}>
          {testing ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Database className="mr-1 h-3 w-3" />}
          Test Connection
        </Button>
        {dbTestResult && (
          dbTestResult.ok
            ? <span className="inline-flex items-center gap-1 text-xs text-green-600 dark:text-green-400"><CheckCircle className="h-3.5 w-3.5" /> Connected</span>
            : <span className="inline-flex items-center gap-1 text-xs text-red-600 dark:text-red-400"><XCircle className="h-3.5 w-3.5" /> {dbTestResult.error}</span>
        )}
      </div>

      <div className="flex items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={handleMigrate}
          disabled={migrating || !dbTestResult?.ok}
        >
          {migrating ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <HardDrive className="mr-1 h-3 w-3" />}
          Run Migrations
        </Button>
        {migrateResult && (
          migrateResult.ok
            ? <span className="inline-flex items-center gap-1 text-xs text-green-600 dark:text-green-400">
                <CheckCircle className="h-3.5 w-3.5" />
                Migrations applied{migrateResult.revision ? <code className="ml-1 rounded bg-green-500/10 px-1 py-0.5 font-mono text-[10px]">rev {migrateResult.revision}</code> : null}
              </span>
            : <span className="inline-flex items-center gap-1 text-xs text-red-600 dark:text-red-400"><XCircle className="h-3.5 w-3.5" /> {migrateResult.error}</span>
        )}
      </div>

      <SaveButton onClick={handleSave} saving={saving} saved={saved} />
    </div>
  );
}

function RetrievalTab({ settings }: { settings: UseSettingsReturn }) {
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

function LLMTab({ settings }: { settings: UseSettingsReturn }) {
  const { config, updateSection, saving, restartRequired } = settings;
  const [llm, setLlm] = useState({ primary: "cody", fallback: "ollama" });
  const [cody, setCody] = useState({ model: "", endpoint: "", access_token_env: "SRC_ACCESS_TOKEN" });
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
        </div>

        {/* Test Connection */}
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={handleTestCody} disabled={codyTest.loading}>
            {codyTest.loading ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Brain className="mr-1 h-3 w-3" />}
            Test Connection
          </Button>
          {codyTest.result && !codyTest.result.ok && (
            <span className="inline-flex items-center gap-1 text-xs text-red-600 dark:text-red-400">
              <XCircle className="h-3.5 w-3.5" /> {codyTest.result.error}
            </span>
          )}
        </div>

        {/* Auth success display */}
        {codyTest.result?.ok && codyTest.result.user && (
          <div className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400">
            <CheckCircle className="h-3.5 w-3.5 shrink-0" />
            <span>
              Authenticated as <strong>{codyTest.result.user.displayName}</strong>
              {codyTest.result.user.email && (
                <span className="text-muted-foreground"> ({codyTest.result.user.email})</span>
              )}
            </span>
          </div>
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
            {ollamaTest.loading ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <HardDrive className="mr-1 h-3 w-3" />}
            Test Connection
          </Button>
          {ollamaTest.result && !ollamaTest.result.ok && (
            <span className="inline-flex items-center gap-1 text-xs text-red-600 dark:text-red-400">
              <XCircle className="h-3.5 w-3.5" /> {ollamaTest.result.error}
            </span>
          )}
        </div>

        {/* Success display */}
        {ollamaTest.result?.ok && (
          <div className="flex items-center gap-1.5 text-xs text-green-600 dark:text-green-400">
            <CheckCircle className="h-3.5 w-3.5 shrink-0" />
            <span>Connected &mdash; {ollamaTest.result.models?.length ?? 0} models available</span>
          </div>
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

function IntelligenceTab({ settings }: { settings: UseSettingsReturn }) {
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

function GenerationTab({ settings }: { settings: UseSettingsReturn }) {
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

function VerificationTab({ settings }: { settings: UseSettingsReturn }) {
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

function IndexingTab({ settings, indexing }: { settings: UseSettingsReturn; indexing: UseIndexingReturn }) {
  const { config, updateSection, saving, restartRequired } = settings;
  const [repoPath, setRepoPath] = useState("");
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
    <div className="space-y-5">
      {/* ── Repo path + Start/Cancel ─────────────────────────── */}
      <div className="space-y-2">
        <Label htmlFor="repo-path">Documentation Repository Path</Label>
        <div className="flex gap-2">
          <div className="relative flex-1">
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
          Full path to a documentation folder. A new index will replace the old one.
        </p>
      </div>

      {/* ── Progress section ──────────────────────────────────── */}
      {isRunning && (
        <div className="space-y-3 rounded-lg border border-blue-500/30 bg-blue-500/5 p-3 dark:bg-blue-500/10">
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
            <ScrollArea className="h-24 rounded border border-border/50 bg-background/50 p-2">
              <div className="space-y-0.5">
                {indexing.recentFiles.map((f, i) => (
                  <div
                    key={`${f.file}-${i}`}
                    className="flex items-center gap-1.5 text-[11px] text-muted-foreground"
                  >
                    <FileText className="h-3 w-3 shrink-0" />
                    <span className="truncate">{f.file.split("/").pop()}</span>
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
          <div>
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

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-h-[85vh] max-w-[620px] overflow-y-auto">
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
                className="mt-2"
              >
                <TabsList className="flex-wrap">
                  <TabsTrigger value="database" className="gap-1 text-xs" disabled={tour.active}>
                    <Database className="h-3.5 w-3.5" /> Database
                  </TabsTrigger>
                  <TabsTrigger value="retrieval" className="gap-1 text-xs" disabled={tour.active}>
                    <Search className="h-3.5 w-3.5" /> Retrieval
                  </TabsTrigger>
                  <TabsTrigger value="llm" className="gap-1 text-xs" disabled={tour.active}>
                    <Brain className="h-3.5 w-3.5" /> LLM{tabBadge(["llm", "cody", "ollama"])}
                  </TabsTrigger>
                  <TabsTrigger value="intelligence" className="gap-1 text-xs" disabled={tour.active}>
                    <Sparkles className="h-3.5 w-3.5" /> Intel
                  </TabsTrigger>
                  <TabsTrigger value="generation" className="gap-1 text-xs" disabled={tour.active}>
                    <Sparkles className="h-3.5 w-3.5" /> Gen
                  </TabsTrigger>
                  <TabsTrigger value="verification" className="gap-1 text-xs" disabled={tour.active}>
                    <ShieldCheck className="h-3.5 w-3.5" /> Verify
                  </TabsTrigger>
                  <TabsTrigger value="indexing" className="gap-1 text-xs" disabled={tour.active}>
                    <Radio className="h-3.5 w-3.5" /> Index{tabBadge(["indexing"])}
                  </TabsTrigger>
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

            {/* Footer — "Take a Tour" button */}
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

import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Database, HardDrive, Loader2, CheckCircle, XCircle } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import type { UseSettingsReturn } from "@/hooks/use-settings";
import { field, SaveButton, AnimatedCheckmark, LoadingDots } from "./shared";

export function DatabaseTab({ settings }: { settings: UseSettingsReturn }) {
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
                    {testing ? <LoadingDots /> : <Database className="mr-1 h-3 w-3" />}
                    {!testing && "Test Connection"}
                </Button>
                <AnimatePresence mode="wait">
                    {dbTestResult && (
                        dbTestResult.ok
                            ? <motion.span key="ok" initial={{ opacity: 0, x: -4 }} animate={{ opacity: 1, x: 0 }} className="inline-flex items-center gap-1 text-xs text-green-600 dark:text-green-400"><AnimatedCheckmark /> Connected</motion.span>
                            : <motion.span key="err" initial={{ opacity: 0, x: -4 }} animate={{ opacity: 1, x: 0 }} className="inline-flex items-center gap-1 text-xs text-red-600 dark:text-red-400"><XCircle className="h-3.5 w-3.5" /> {dbTestResult.error}</motion.span>
                    )}
                </AnimatePresence>
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

import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";
import { cn } from "../../lib/cn";

export function Button({
  className,
  variant = "primary",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" | "outline" | "danger" }) {
  const styles = {
    primary: "bg-accent text-white hover:bg-accent/90",
    ghost: "hover:bg-panel-hover text-muted",
    outline: "border border-line hover:bg-panel-hover",
    danger: "bg-danger/15 text-danger hover:bg-danger/25",
  };
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition disabled:opacity-40",
        styles[variant],
        className
      )}
      {...props}
    />
  );
}

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        "w-full rounded-md border border-line bg-input px-3 py-2 text-sm outline-none focus:border-accent",
        props.className
      )}
    />
  );
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={cn(
        "rounded-md border border-line bg-panel px-2 py-1.5 text-sm text-fg outline-none focus:border-accent",
        props.className
      )}
    />
  );
}

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("rounded-lg border border-line bg-panel", className)}>{children}</div>;
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "ok" | "warn" | "accent";
}) {
  const map = {
    neutral: "bg-line/50 text-muted",
    ok: "bg-ok/15 text-ok",
    warn: "bg-warn/15 text-warn",
    accent: "bg-accent/15 text-accent",
  };
  return (
    <span className={cn("inline-flex items-center rounded px-2 py-0.5 font-mono text-[11px] uppercase tracking-wide", map[tone])}>
      {children}
    </span>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block text-xs font-medium text-muted">
      {label}
      <div className="mt-1">{children}</div>
    </label>
  );
}

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        panel: "var(--panel)",
        "panel-hover": "var(--panel-hover)",
        line: "var(--line)",
        fg: "var(--fg)",
        muted: "var(--muted)",
        accent: "var(--accent)",
        warn: "var(--warn)",
        ok: "var(--ok)",
        danger: "var(--danger)",
        input: "var(--input)",
        canvas: "var(--canvas)",
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};

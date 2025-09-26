// api/gorq.js
const fetch = require("node-fetch");

function parseFloatEnv(name, fallback) {
  const v = process.env[name];
  return v !== undefined ? parseFloat(v) : fallback;
}
function parseIntEnv(name, fallback) {
  const v = process.env[name];
  return v !== undefined ? parseInt(v, 10) : fallback;
}
function parseBoolEnv(name, fallback) {
  const v = process.env[name];
  return v !== undefined ? (v === "true" || v === "1") : fallback;
}

module.exports = async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*"); // para testes; em produção restrinja
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();

  try {
    const API_KEY = process.env.GROQ_API_KEY;
    if (!API_KEY) return res.status(500).json({ error: "GROQ_API_KEY não configurada." });

    // Defaults from env
    const defaults = {
      model: process.env.GROQ_MODEL || "openai/gpt-oss-20b",
      name: process.env.AI_NAME || "IA",
      persona: process.env.AI_PERSONA || `Você é ${process.env.AI_NAME || 'IA'}, responda de forma clara e útil.`,
      temperature: parseFloatEnv("AI_TEMPERATURE", 0.7),
      max_tokens: parseIntEnv("AI_MAX_TOKENS", 400),
      top_p: parseFloatEnv("AI_TOP_P", 1),
      presence_penalty: parseFloatEnv("AI_PRESENCE_PEN", 0.0),
      frequency_penalty: parseFloatEnv("AI_FREQUENCY_PEN", 0.0),
      lang: process.env.AI_LANG || "pt-BR",
      tone: process.env.AI_TONE || "neutro",
      response_fmt: process.env.AI_RESPONSE_FMT || "text",
      safe_mode: parseBoolEnv("AI_SAFE_MODE", true)
    };

    // Allow request to override defaults: { prompt, overrides: { temperature, max_tokens, name, persona... } }
    const body = req.body && typeof req.body === "object" ? req.body :
                 (req.body ? JSON.parse(req.body) : {});
    const prompt = body.prompt || body.promptText || "";
    if (!prompt) return res.status(400).json({ error: "Prompt é obrigatório." });

    const overrides = body.overrides || {};
    const cfg = {
      model: overrides.model || defaults.model,
      name: overrides.name || defaults.name,
      persona: overrides.persona || defaults.persona,
      temperature: overrides.temperature !== undefined ? overrides.temperature : defaults.temperature,
      max_tokens: overrides.max_tokens !== undefined ? overrides.max_tokens : defaults.max_tokens,
      top_p: overrides.top_p !== undefined ? overrides.top_p : defaults.top_p,
      presence_penalty: overrides.presence_penalty !== undefined ? overrides.presence_penalty : defaults.presence_penalty,
      frequency_penalty: overrides.frequency_penalty !== undefined ? overrides.frequency_penalty : defaults.frequency_penalty,
      lang: overrides.lang || defaults.lang,
      tone: overrides.tone || defaults.tone,
      response_fmt: overrides.response_fmt || defaults.response_fmt,
      safe_mode: overrides.safe_mode !== undefined ? overrides.safe_mode : defaults.safe_mode
    };

    // Build messages array (system preamble + user prompt)
    const systemMsg = `${cfg.persona}
-- Regras rápidas:
- Fale em ${cfg.lang}.
- Tom: ${cfg.tone}.
- Seja conciso se o pedido pedir.
- Se pedido contiver ação insegura ou ilegal, recuse.
`;
    const messages = [
      { role: "system", content: systemMsg },
      { role: "user", content: prompt }
    ];

    // Optional: you can attach conversation memory here if you store it somewhere.

    // Build request body to Groq/OpenAI-compatible endpoint
    const payload = {
      model: cfg.model,
      messages,
      temperature: cfg.temperature,
      max_tokens: cfg.max_tokens,
      top_p: cfg.top_p,
      presence_penalty: cfg.presence_penalty,
      frequency_penalty: cfg.frequency_penalty,
      // stop: overrides.stop || undefined
    };

    // Call Groq
    const r = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload),
    });

    const data = await r.json();

    // Normalize output: try to extract assistant text
    let content = null;
    if (Array.isArray(data?.choices) && data.choices[0]?.message?.content) {
      content = data.choices[0].message.content;
    } else if (Array.isArray(data?.choices) && data.choices[0]?.text) {
      content = data.choices[0].text;
    }

    // If safe_mode is ON, you can optionally post-process content here (ex: remove disallowed words)
    if (cfg.safe_mode && typeof content === "string") {
      // simple example: strip suspicious instructions — you can expand rules
      const banned = ["bomba", "como fabricar"];
      banned.forEach(w => {
        const re = new RegExp(w, "gi");
        content = content.replace(re, "[conteúdo removido]");
      });
    }

    return res.status(200).json({
      name: cfg.name,
      content: content,
      raw: data
    });
  } catch (err) {
    console.error("Erro interno /gorq:", err);
    return res.status(500).json({ error: "Erro interno", details: (err && err.message) || err });
  }
};

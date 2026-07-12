const COLOR_HARMONY_SYSTEM_PROMPT = '你是配色分析助手。判断给定聚类颜色均值之间的搭配是否和谐，用中文简短回答。';

function buildColorHarmonyPrompt(palette) {
    const colors = palette.map((p, index) =>
        `c${index + 1}: ${p.hex}, RGB(${p.r}, ${p.g}, ${p.b}), 占比${p.percentage}`
    ).join('\n');

    return `请判断以下颜色组合是否和谐，并说明主要原因，最后给出“结论：和谐/一般/不和谐”。\n${colors}`;
}

async function analyzeColorHarmonyWithAI({ apiKey, endpoint, model, palette }) {
    const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + apiKey
        },
        body: JSON.stringify({
            model,
            messages: [
                { role: 'system', content: COLOR_HARMONY_SYSTEM_PROMPT },
                { role: 'user', content: buildColorHarmonyPrompt(palette) }
            ],
            temperature: 0.3
        })
    });

    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || ('HTTP ' + response.status));
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content || '未返回分析结果';
}

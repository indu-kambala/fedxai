import type { CaseRecord } from "./types";

const SUGGESTED = [
  "Why did the model make this prediction?",
  "Which regions influenced the prediction?",
  "What evidence supports this interpretation?",
  "What evidence contradicts it?",
  "How does this compare with COVID-19?",
  "What additional tests might help distinguish these conditions?",
  "Show relevant literature.",
  "Explain this in simple terms.",
];

export const suggestedQuestions = SUGGESTED;

function stamp() {
  return new Date().toISOString();
}

export function groundedReply(question: string, current: CaseRecord | null, modelName: string) {
  const q = question.toLowerCase();
  const sources = [
    {
      title: "FedXAI-Med clinical guardrails",
      source: "Project documentation",
      excerpt:
        "The assistant must separate model output, image explanation, retrieved evidence, general knowledge, and uncertainty. It must not invent findings.",
    },
  ];

  if (!current?.predictedClass) {
    return {
      content: [
        "No model prediction is attached to this case yet.",
        "The inference service is not connected in this build, so I cannot assign NORMAL, PNEUMONIA, or COVID-19, and I will not invent a class or confidence score.",
        "Upload a chest X-ray in the analysis workspace, then treat any future model output as AI-assisted decision support that requires clinician review.",
        current
          ? `A local image is loaded (${current.fileName}) using ${modelName}. Visual review is available; classification is pending.`
          : "Open New Case to load a radiograph first.",
      ].join("\n\n"),
      citations: sources,
    };
  }

  if (q.includes("simple")) {
    return {
      content: `The configured model (${modelName}) predicted ${current.predictedClass} with a reported model probability of ${((current.modelProbability ?? 0) * 100).toFixed(1)}%. That number is not diagnostic certainty. A qualified clinician must review the image and the clinical context.`,
      citations: sources,
    };
  }

  if (q.includes("region") || q.includes("why") || q.includes("heatmap") || q.includes("attribution")) {
    return {
      content: `The model predicted ${current.predictedClass}. Attribution overlays, when generated, show where the network placed weight — they are not radiographic findings and do not prove disease. ${current.notes || "No attribution map has been computed for this case yet."}`,
      citations: sources,
    };
  }

  if (q.includes("contradict") || q.includes("covid")) {
    return {
      content:
        "Chest radiographs for pneumonia and COVID-19 can overlap. Distinguishing them requires clinical history, laboratory data, and often additional imaging. This assistant will not claim that a heatmap proves one infection over the other. Retrieved literature will be listed only when the knowledge base returns a document.",
      citations: sources,
    };
  }

  if (q.includes("literature") || q.includes("evidence") || q.includes("support")) {
    return {
      content:
        "No indexed PubMed corpus is connected yet. I will not fabricate papers. When RAG is enabled, answers will include document title, excerpt, and citation. Until then, only project documentation is available as a source.",
      citations: sources,
    };
  }

  if (q.includes("test") || q.includes("additional")) {
    return {
      content:
        "Additional information that often helps clinicians (not ordered by this system): symptom timeline, oxygen saturation, inflammatory markers, PCR or antigen testing where appropriate, comparison with prior radiographs, and the original clinical question. This is general knowledge, not a care plan.",
      citations: sources,
    };
  }

  return {
    content: `The model predicted ${current.predictedClass}. Calibration status: ${current.calibrationStatus}. ${current.uncertainty} AI confidence is not equivalent to diagnostic certainty. This requires clinical verification.`,
    citations: sources,
    createdAt: stamp(),
  };
}

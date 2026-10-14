export type Role = "doctor" | "researcher" | "hospital_admin" | "system_admin";

export const HOSPITALS = [
  { id: "hospital_a", name: "Hospital A" },
  { id: "hospital_b", name: "Hospital B" },
  { id: "hospital_c", name: "Hospital C" },
  { id: "hospital_d", name: "Hospital D" },
  { id: "hospital_e", name: "Hospital E" },
] as const;

export const MODELS = [
  { id: "densenet121", name: "DenseNet-121", family: "cnn" },
  { id: "efficientnet_b3", name: "EfficientNet-B3", family: "cnn" },
  { id: "vit_b16", name: "ViT-B/16", family: "transformer" },
  { id: "swin_tiny", name: "Swin Transformer", family: "transformer" },
] as const;

export const CLASSES = ["NORMAL", "PNEUMONIA", "COVID-19"] as const;

export const DISCLAIMER =
  "AI-assisted research prototype. Predictions require review by a qualified clinician and are not a medical diagnosis.";

export type CaseRecord = {
  id: string;
  createdAt: string;
  hospitalId: string;
  modelId: string;
  fileName: string;
  imageDataUrl: string;
  predictedClass: string | null;
  modelProbability: number | null;
  calibrationStatus: string;
  uncertainty: string;
  notes: string;
};

export type ChatMessage = {
  id: string;
  role: "doctor" | "assistant";
  content: string;
  createdAt: string;
  citations?: Array<{ title: string; source: string; excerpt: string }>;
};

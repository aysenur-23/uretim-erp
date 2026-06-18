export type Thresholds = {
  crackSensitivity: number;
  burnDeltaV: number;
  rawDeltaV: number;
  rawDeltaS: number;
  warnScore: number;
  redScore: number;
};

export const DEFAULT_THRESHOLDS: Thresholds = {
  crackSensitivity: 16,
  burnDeltaV: 22,
  rawDeltaV: 14,
  rawDeltaS: 8,
  warnScore: 40,
  redScore: 70,
};

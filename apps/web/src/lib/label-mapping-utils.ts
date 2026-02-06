import type { Label, DetectionLabelMapping, DetectionLabelMappingConfig } from '@/types/annotations'

/**
 * Generate a localStorage key for persisting label mapping
 */
export function getLabelMappingStorageKey(modelId: string, projectContext: string): string {
  return `autodetect-mapping:${modelId}:${projectContext}`
}

/**
 * Save label mapping to localStorage
 */
export function saveLabelMapping(key: string, mapping: DetectionLabelMappingConfig): void {
  try {
    localStorage.setItem(key, JSON.stringify(mapping))
  } catch {
    // localStorage full or unavailable - silently fail
  }
}

/**
 * Load label mapping from localStorage
 */
export function loadLabelMapping(key: string): DetectionLabelMappingConfig | null {
  try {
    const stored = localStorage.getItem(key)
    if (!stored) return null
    return JSON.parse(stored) as DetectionLabelMappingConfig
  } catch {
    return null
  }
}

/**
 * Build a default mapping by auto-matching model classes to project labels by name (case-insensitive).
 * Unmatched classes default to 'skip'.
 */
export function buildDefaultMapping(
  modelClasses: string[],
  projectLabels: Label[]
): DetectionLabelMappingConfig {
  const mapping: DetectionLabelMappingConfig = {}

  for (const cls of modelClasses) {
    const matchedLabel = projectLabels.find(
      (l) => l.name.toLowerCase() === cls.toLowerCase()
    )

    if (matchedLabel) {
      mapping[cls] = { action: 'map', projectLabelId: matchedLabel.id }
    } else {
      mapping[cls] = { action: 'skip' }
    }
  }

  return mapping
}

/**
 * Resolve per-detection label IDs from model labels using the mapping.
 * Returns an array where each element is either a project label ID or 'skip'.
 * If modelLabels is empty/undefined, returns fallbackLabelId for all detections.
 */
export function resolveAllLabels(
  modelLabels: string[] | undefined,
  mapping: DetectionLabelMappingConfig,
  fallbackLabelId: string | null,
  count: number
): Array<string | 'skip'> {
  if (!modelLabels || modelLabels.length === 0) {
    // No per-detection labels from model - use fallback for all
    return Array(count).fill(fallbackLabelId || 'skip')
  }

  return modelLabels.map((className) => {
    const entry: DetectionLabelMapping | undefined = mapping[className]
    if (!entry || entry.action === 'skip') return 'skip'
    if (entry.action === 'map' && entry.projectLabelId) return entry.projectLabelId
    return fallbackLabelId || 'skip'
  })
}

/**
 * Filter detection results to remove skipped detections.
 * Returns filtered arrays and the resolved label IDs.
 */
export function filterSkippedDetections<T>(
  items: T[],
  resolvedLabels: Array<string | 'skip'>
): { filtered: T[]; labelIds: string[] } {
  const filtered: T[] = []
  const labelIds: string[] = []

  for (let i = 0; i < items.length; i++) {
    const label = resolvedLabels[i]
    if (label !== 'skip') {
      filtered.push(items[i])
      labelIds.push(label)
    }
  }

  return { filtered, labelIds }
}

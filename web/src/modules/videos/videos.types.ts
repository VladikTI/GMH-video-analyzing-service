import * as v from 'valibot';

const keyIntervalSchema = v.object({
  start_time: v.string(),
  end_time: v.string(),
  title: v.string(),
  objects: v.array(v.string()),
  events: v.array(v.string()),
  sounds: v.array(v.string()),
});

const tagsSchema = v.object({
  category: v.string(),
  objects: v.optional(v.array(v.string())),
  events: v.optional(v.array(v.string())),
  music: v.optional(v.array(v.string())),
  sounds: v.optional(v.array(v.string())),
  symbols: v.optional(v.array(v.string())),
  voice_transcription: v.string(),
});

export const videoDetailsSchema = v.object({
  key_intervals: v.array(keyIntervalSchema),
  tags: tagsSchema,
  tonality: v.record(v.string(), v.array(v.string())),
});

export type VideoDetails = v.InferInput<typeof videoDetailsSchema>;


export const videoInfoSchema = v.object({
  file_id: v.number(),
  status: v.string(),
  title: v.string(),
  details: v.optional(videoDetailsSchema),
});

export type VideoInfo = v.InferInput<typeof videoInfoSchema>;

export interface VideoTag {
  name: string;
}

import { VideoDetails, videoDetailsSchema } from '../modules/videos/videos.types';
import api from './base';
import * as v from 'valibot';

const resSchema = v.object({
  id: v.string(),
  details: videoDetailsSchema,
})

export interface LoadVideoDetailsOpts {
  id: string;
}

export type LoadVideoDetailsRes = v.InferInput<typeof resSchema>;

export const loadVideoDetailsRequest = async (opts: LoadVideoDetailsOpts) => {
  try {
    const res = await api.get<VideoDetails>('/details',{
      params: opts
    });
    return v.parse(resSchema, res.data);
  } catch (err: any) {

    throw err.response.data.error;
  }
};

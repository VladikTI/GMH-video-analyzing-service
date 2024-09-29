import api from './base';
import { VideoInfo, videoInfoSchema } from '../modules/videos/videos.types';
import * as v from 'valibot';

export const loadVideosInfoRequest = async () => {
  try {
    const res = await api.get<VideoInfo[]>('/videos');
    return v.parse(v.array(videoInfoSchema), res.data);
  } catch (err: any) {
    throw err.response.data.error;
  }
};

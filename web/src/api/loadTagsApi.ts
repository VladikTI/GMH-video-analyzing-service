import { VideoTag } from '../modules/videos/videos.types';
import api from './base';

export const loadTagsRequest = async () => {
  try {
    const res = await api.get<{tags: VideoTag[]}>('/tags');
    return res.data.tags;
  } catch (err: any) {

    throw err.response.data.error;
  }
};

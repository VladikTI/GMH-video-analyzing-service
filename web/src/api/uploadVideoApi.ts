import { AxiosProgressEvent } from 'axios';
import api from './base';

type uploadProgressHandler = (e: AxiosProgressEvent) => void;

export interface uploadVideoOpts {
  data: FormData;
}

export const uploadVideoRequest = async (
  opts: uploadVideoOpts,
  onUploadProgress: uploadProgressHandler
) => {
  const { data } = opts;

  try {
    const result = await api.post('/upload', data, {  onUploadProgress });

    console.log('result is', result); // result is server's response
  } catch (error) {
    console.error(error);
  } finally {
    console.log('Upload complete');
  }
};

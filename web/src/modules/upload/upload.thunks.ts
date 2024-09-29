import { createAsyncThunk } from '@reduxjs/toolkit';
import { uploadVideoOpts, uploadVideoRequest } from '../../api/uploadVideoApi';
import { AxiosProgressEvent } from 'axios';
import { setUploadProgress } from './upload.slice';

export const uploadVideo = createAsyncThunk<
  void,
  uploadVideoOpts,
  { rejectValue: string }
>('upload/uploadVideo', async (opts, thunkApi) => {
  const { dispatch } = thunkApi;

  const onUploadProgress = (event: AxiosProgressEvent) => {
    const percentCompleted = Math.round(
      (event.loaded * 100) / (event.total || 0)
    );
    dispatch(setUploadProgress(percentCompleted));
  };

  await uploadVideoRequest(opts, onUploadProgress);
});

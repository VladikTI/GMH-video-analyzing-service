import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { uploadVideo } from './upload.thunks';

interface InitialState {
  error: string | null;
  progress: number;
  uploading: boolean;
}

const initialState: InitialState = {
  error: null,
  progress: 0,
  uploading: false,
};

export const uploadSlice = createSlice({
  name: 'upload',
  initialState,
  reducers: {
    setUploadProgress: (state, action: PayloadAction<number>) => {
      state.progress = action.payload;
    }
  },
  selectors: {
    uploadProgress: (state) => state.progress,
    isUploading: (state) => state.uploading,
  },
  extraReducers: (builder) => {
    builder
    .addCase(uploadVideo.pending, (state) => {
      state.uploading = true;
      state.error = null;
      state.progress = 0;
    })
    .addCase(uploadVideo.fulfilled, (state) => {
      state.uploading = false;
      state.error = null;
    })
    .addCase(uploadVideo.rejected, (state, action) => {
      state.uploading = false;
      state.error = action.payload || null;
    })
  },
});

export const { isUploading, uploadProgress} = uploadSlice.selectors;

export const { setUploadProgress } = uploadSlice.actions;

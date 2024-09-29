import { combineSlices, configureStore } from '@reduxjs/toolkit';
import { useDispatch, useSelector } from 'react-redux';
import { videosSlice } from '../modules/videos/videos.slice';
import { uploadSlice } from '../modules/upload/upload.slice';

export const store = configureStore({
  reducer: combineSlices(videosSlice, uploadSlice),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export const useAppSelector = useSelector.withTypes<RootState>();
export const useAppDispatch = useDispatch.withTypes<AppDispatch>();

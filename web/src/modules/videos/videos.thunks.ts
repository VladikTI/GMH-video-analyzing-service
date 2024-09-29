import { createAsyncThunk } from '@reduxjs/toolkit';
import WebsocketService from '../../services/websocketService';
import { setSocket, setVideosInfo } from './videos.slice';
import { RootState } from '../../store';

export const loadVideosInfo = createAsyncThunk<
  void
>('videos/loadVideosInfo', async (_, thunkApi) => {
    const videosSocket = new WebsocketService({
      url: `/api/videos/list`,
      callbacks: {
        onClose: () => {},
        onError: () => {},
        onMessage: (data) => {
          thunkApi.dispatch(setVideosInfo(data));
        },
        onOpen: () => {
          thunkApi.dispatch(setSocket(videosSocket));
        },
        onDisconnect:  () => {
          thunkApi.dispatch(setSocket(null));
        },
      },
    });
});

export const disconnectSocket = createAsyncThunk<void, void, {state: RootState}>('videos/disconnectSocket', async (_, thunkApi) => {
  const {getState} = thunkApi;
  getState().videos.socket?.disconnect();
})

export const loadTags = createAsyncThunk<
  void,
  void,
  {
    state: RootState
  }
>('videos/loadTags', async (_, thunkApi) => {
  try {
    thunkApi.getState().videos.socket?.send({type: 'filters', data: []})
  } catch (err: any) {
    return thunkApi.rejectWithValue(err);
  }
});


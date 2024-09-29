import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { VideoInfo } from './videos.types';
import WebsocketService from '../../services/websocketService';

interface InitialState {
  error: string | null;
  loading: boolean;
  data: VideoInfo[];
  filters: {name: string, active: boolean}[];
  socket: WebsocketService | null;
}

const initialState: InitialState = {
  error: null,
  loading: false,
  data: [],
  filters: ['Авто', 'Блогеры', 'Видеоигры', 'Детям', 'Еда', 'Спорт', 'Культура', 'Лайфхаки', 'Музыка', 'Новости и СМИ', 'Обучение',
    'Аниме', 'Интервью', 'Подкасты', 'Путешествия', 'Радио', 'Сериалы', 'Телешоу', 'Фильмы', 'ТВ онлайн', 'Трансляции', 'Телеканалы', 'Футбол', 'Юмор']
    .map((item) => ({name: item, active: false})),
  socket: null,
};

export const videosSlice = createSlice({
  name: 'videos',
  initialState,
  reducers: {
    addVideoFilter: (state, action: PayloadAction<string>) => {
      state.filters = state.filters.map((item) => {
        if (item.name === action.payload) {
          return {name: item.name, active: true}
        }

        return item;
      })
    },
    removeVideoFilter: (state, action: PayloadAction<string>) => {
      state.filters = state.filters.map((item) => {
        if (item.name === action.payload) {
          return {name: item.name, active: false}
        }

        return item;
      })
    },
    clearVideoFilters: (state) => {
      state.filters = initialState.filters;
    },
    setVideosInfo: (state, action: PayloadAction<string>) => {
      state.data = JSON.parse(action.payload).data;
    },
    setSocket: (state, action: PayloadAction<WebsocketService | null>) => {
      state.socket = action.payload;
    },
  },
  selectors: {
    videosInfo: (state) => {
      let filtersActive = false;
      const filteredData = state.data.filter((video) => {
        for (const filter of state.filters) {
          if (filter.active) {
            filtersActive = true;
          }
          if (video.details?.tags.category === filter.name && filter.active) {
            return true;
          }
        }
        return false;
      });

      return filtersActive ? filteredData : state.data;
    },
    videoFilters: (state) => state.filters
  },
});

export const { videosInfo, videoFilters } = videosSlice.selectors;

export const {
  addVideoFilter,
  clearVideoFilters,
  removeVideoFilter,
  setVideosInfo,
  setSocket,
} = videosSlice.actions;



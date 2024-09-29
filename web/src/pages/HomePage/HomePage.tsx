import { Button, Card, Modal, TextInput } from '@gravity-ui/uikit';
import VideoFilters from '../../blocks/VideoFilters/VideoFilters';
import { useAppDispatch, useAppSelector } from '../../store';
import { videosInfo } from '../../modules/videos/videos.slice';
import { useEffect, useMemo, useState } from 'react';
import {
  disconnectSocket,
  loadVideosInfo,
} from '../../modules/videos/videos.thunks';
import styles from './HomePage.module.scss';
import UploadVideoModal from '../../blocks/UploadVideoModal/UploadVideoModal';
import VideoListItem from '../../components/VideoListItem/VideoListItem';
import { VideoInfo } from '../../modules/videos/videos.types';
import MiniSearch from 'minisearch';

const miniSearch = new MiniSearch({
  fields: ['voice_transcription'],
  storeFields: ['content_type', 'details', 'file_id', 'status', 'title'],
  searchOptions: {
    fuzzy: 0.2,
  },
});

const HomePage = () => {
  const dispatch = useAppDispatch();
  const data = useAppSelector(videosInfo);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [search, setSearch] = useState('');

  useEffect(() => {
    dispatch(loadVideosInfo());

    return () => {
      dispatch(disconnectSocket);
    };
  }, []);

  useEffect(() => {
    miniSearch.removeAll();
    miniSearch.addAll(
      data.map((item) => ({
        ...item,
        id: item.file_id,
        voice_transcription: item.details?.tags.voice_transcription,
      }))
    );

    return () => {
      miniSearch.removeAll();
    };
  }, [data]);

  const searchedRes = useMemo(() => {
    const res = miniSearch.search(search);
    return res.length ? res : data;
  }, [search, data]);

  return (
    <div className={styles.page}>
      <Button
        className={styles.add}
        onClick={() => setUploadModalOpen(true)}
        size="xl"
      >
        Загрузить видео
      </Button>
      <Modal open={uploadModalOpen} onClose={() => setUploadModalOpen(false)}>
        <UploadVideoModal />
      </Modal>
      <div className={styles.filters}>
        <VideoFilters />
      </div>
      <Card className={styles.list}>
        <TextInput
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Поиск"
        />
        <div className={styles.listWrapper}>
          {searchedRes &&
            Array.from(searchedRes as unknown as VideoInfo[])
              .reverse()
              .map((item) => <VideoListItem item={item} />)}
        </div>
      </Card>
    </div>
  );
};

export default HomePage;

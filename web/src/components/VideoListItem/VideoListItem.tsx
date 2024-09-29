import { Button, Label, Text, Card } from '@gravity-ui/uikit';
import { VideoInfo } from '../../modules/videos/videos.types';

import styles from './VideoListItem.module.scss';
import { useNavigate } from 'react-router-dom';
import { useMemo } from 'react';

interface IProps {
  item: VideoInfo;
}

const VideoListItem = ({ item }: IProps) => {
  const navigate = useNavigate();

  const openVideoDetails = () => {
    navigate(`dashboard/${item.file_id}`);
  };

  const tags = useMemo(() => {
    if (!item.details) {
      return [];
    }
    const {
      objects = [],
      events = [],
      music = [],
      sounds = [],
      symbols = [],
    } = item.details.tags;
    return [
      ...new Set([...objects, ...events, ...music, ...sounds, ...symbols]),
    ];
  }, [item.details]);

  return (
    <Card view="outlined" className={styles.container}>
      <Text variant="header-2">{item.title}</Text>
      <div>
        <Text variant="subheader-2">Статус обработки: {item.status}</Text>
      </div>
      {item.details && (
        <div>
          <Text variant="subheader-2">
            Категория: {item.details.tags.category}
          </Text>
        </div>
      )}
      <div>
        {item.status === 'completed' && (
          <Button onClick={openVideoDetails} size="xl">
            Подробная информация
          </Button>
        )}
      </div>
      {!!item.details?.tags.voice_transcription && (
        <div className={styles.description}>
          <Text>{item.details?.tags.voice_transcription}</Text>
        </div>
      )}
      {!!tags.length && (
        <ul className={styles.list}>
          {tags.map((item) => (
            <li>
              <Label theme="normal">{item}</Label>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
};

export default VideoListItem;

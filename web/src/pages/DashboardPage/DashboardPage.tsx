import { Card, Text } from '@gravity-ui/uikit';
import styles from './DashboardPage.module.scss';
import { useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useAppSelector } from '../../store';
import { videosInfo } from '../../modules/videos/videos.slice';

interface Props {
  start: string;
  end: string;
  onClick: () => void;
  title: string;
}

const TimelineBlock = ({ onClick, title, start, end }: Props) => {
  return (
    <div onClick={onClick} className={styles.timelineBlock}>
      <Text>{title}</Text>
      <Text className={styles.timestamp}>
        {start} - {end}
      </Text>
    </div>
  );
};

const DashboardPage = () => {
  const { id } = useParams();
  const [selectedInterval, setSelectedInterval] = useState<number | null>(0);
  const videos = useAppSelector(videosInfo);

  const details = useMemo(() => {
    if (!id) {
      return undefined;
    }

    const res = videos.find((item) => item.file_id === parseInt(id));
    return res ? res.details : undefined;
  }, [videos, id]);

  return (
    <Card className={styles.container}>
      {details && (
        <>
          <div className={styles.generalInfo}>
            <Text variant="header-1">{details.tags.category}</Text>
            <ul>
              {details.tags.objects &&
                details.tags.objects.map((item) => {
                  return <li>{item}</li>;
                })}
            </ul>
            <div className={styles.description}>
              <Text>{details.tags.voice_transcription}</Text>
            </div>
          </div>
          <div className={styles.timeline}>
            {details.key_intervals &&
              details.key_intervals.map((interval, index) => (
                <TimelineBlock
                  onClick={() => {
                    setSelectedInterval(index);
                  }}
                  start={interval.start_time}
                  end={interval.end_time}
                  title={interval.title}
                />
              ))}
          </div>
          <div className={styles.intervalDetails}>
            {selectedInterval !== null && details.key_intervals && (
              <>
                <Text variant="header-1">
                  {details.key_intervals[selectedInterval].title}
                </Text>
                <div className={styles.tags}>
                  <Card view="raised" className={styles.tagBlock}>
                    <Text variant="subheader-1">События</Text>
                    {details.key_intervals[selectedInterval].events &&
                      details.key_intervals[selectedInterval].events.map(
                        (event) => {
                          return <div>{event}</div>;
                        }
                      )}
                  </Card>
                  <Card view="raised" className={styles.tagBlock}>
                    <Text variant="subheader-1">Объекты</Text>
                    {details.key_intervals[selectedInterval].objects &&
                      details.key_intervals[selectedInterval].objects.map(
                        (object) => {
                          return <div>{object}</div>;
                        }
                      )}
                  </Card>
                  <Card view="raised" className={styles.tagBlock}>
                    <Text variant="subheader-1">Звуки</Text>
                    {details.key_intervals[selectedInterval].sounds &&
                      details.key_intervals[selectedInterval].sounds.map(
                        (sound) => {
                          return <div>{sound}</div>;
                        }
                      )}
                  </Card>
                </div>
              </>
            )}
          </div>
        </>
      )}
    </Card>
  );
};
export default DashboardPage;

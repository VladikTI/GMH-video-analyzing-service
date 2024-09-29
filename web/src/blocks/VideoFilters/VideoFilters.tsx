import FilterOption from '../../components/FilterOption/FilterOption';
import styles from './VideoFilters.module.scss';
import { useEffect } from 'react';
import { useAppDispatch, useAppSelector } from '../../store';
import { loadTags } from '../../modules/videos/videos.thunks';
import { videoFilters } from '../../modules/videos/videos.slice';

const VideoFilters = () => {
  const dispatch = useAppDispatch();
  const filters = useAppSelector(videoFilters);

  useEffect(() => {
    dispatch(loadTags());
  }, []);

  return (
    <div>
      <ul className={styles.list}>
        {filters.map((item) => (
          <FilterOption item={item.name} key={item.name} checked={item.active}/>
        ))}
      </ul>
    </div>
  );
};
export default VideoFilters;

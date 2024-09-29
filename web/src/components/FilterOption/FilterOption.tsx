import { Checkbox, Text } from '@gravity-ui/uikit';
import styles from './FilterOption.module.scss';
import { useAppDispatch } from '../../store';
import {
  addVideoFilter,
  removeVideoFilter,
} from '../../modules/videos/videos.slice';
import { ChangeEvent } from 'react';

interface IProps {
  checked: boolean;
  item: string;
}

const FilterOption = ({ item, checked }: IProps) => {
  const dispatch = useAppDispatch();

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.checked) {
      dispatch(addVideoFilter(item));
    } else {
      dispatch(removeVideoFilter(item));
    }
  };
  return (
    <div className={styles.wrapper}>
      <Checkbox className={styles.checkbox} onChange={handleChange} checked={checked}/>
      <Text>{item}</Text>
    </div>
  );
};

export default FilterOption;

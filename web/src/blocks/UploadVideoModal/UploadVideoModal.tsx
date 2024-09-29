import { Progress, Text } from '@gravity-ui/uikit';
import styles from './UploadVideoModal.module.scss';
import { useState } from 'react';
import { useAppDispatch, useAppSelector } from '../../store';
import { isUploading, uploadProgress } from '../../modules/upload/upload.slice';
import clsx from 'clsx';
import { uploadVideo } from '../../modules/upload/upload.thunks';

const UploadVideoModal = () => {
  const dispatch = useAppDispatch();
  const uploading = useAppSelector(isUploading);
  const progress = useAppSelector(uploadProgress);

  const [drop, setDrop] = useState(false);

  const onDragLeave = (e: React.DragEvent<HTMLElement>) => {
    if (uploading) return;
    e.preventDefault();
    setDrop(false);
  };

  const onDragOver = (e: React.DragEvent<HTMLElement>) => {
    if (uploading) return;
    e.preventDefault();
    setDrop(true);
  };

  const handleDrop = (e: React.DragEvent<HTMLElement>) => {
    if (uploading) return;
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    setDrop(false);

    handleFile(droppedFile);
  };

  const handleFile = (file: File) => {
    if (!file) return;
    const data = new FormData();

    data.append('file', file);

    dispatch(uploadVideo({ data }));
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files) {
      return;
    }

    handleFile(event.target.files[0]);
  };

  return (
    <div className={styles.container}>
      <Text variant="header-2">Добавьте видео</Text>
      {uploading && <Progress text={`${progress}%`} value={progress} />}
      <label
        className={clsx(styles.dndZone, {
          [styles.hover]: drop,
        })}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={handleDrop}
      >
        Перетащите видео сюда или нажмите, чтобы добавить
        <input
          className={styles.input}
          type="file"
          onChange={handleFileChange}
        />
      </label>
    </div>
  );
};

export default UploadVideoModal;

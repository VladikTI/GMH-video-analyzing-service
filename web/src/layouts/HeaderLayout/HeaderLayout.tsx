import { Button } from '@gravity-ui/uikit';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';

import styles from './HeaderLayout.module.scss';

const HeaderLayout = () => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <>
      <nav className={styles.nav}>
        {location.pathname !== '/' && (
          <Button
            className={styles.backButton}
            size="xl"
            onClick={() => navigate('/')}
          >
            На главную
          </Button>
        )}
      </nav>
      <Outlet />
    </>
  );
};
export default HeaderLayout;

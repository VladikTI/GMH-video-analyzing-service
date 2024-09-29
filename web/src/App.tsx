import './index.scss';
import '@gravity-ui/uikit/styles/fonts.css';
import '@gravity-ui/uikit/styles/styles.css';

import { ThemeProvider } from '@gravity-ui/uikit';
import router from './router';
import { RouterProvider } from 'react-router-dom';

const App = () => {
  return (
    <ThemeProvider theme="light">
      <RouterProvider router={router} />
    </ThemeProvider>
  );
};

export default App;

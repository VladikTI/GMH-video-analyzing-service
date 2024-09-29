import { createBrowserRouter } from 'react-router-dom';
import HomePage from './pages/HomePage/HomePage';
import DashboardPage from './pages/DashboardPage/DashboardPage';
import HeaderLayout from './layouts/HeaderLayout/HeaderLayout';

const router = createBrowserRouter([
  {
    path: '/',
    element: <HeaderLayout />,
    children: [
      { path: '/', element: <HomePage /> },
      { path: '/dashboard/:id', element: <DashboardPage /> },
    ],
  },
]);

export default router;

import React from 'react';
import { TopNavbar } from './TopNavbar';

export interface HeaderProps extends React.ComponentProps<typeof TopNavbar> {}

export const Header: React.FC<HeaderProps> = (props) => {
  return <TopNavbar {...props} />;
};

export default Header;

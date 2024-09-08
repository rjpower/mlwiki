'use client'

import { AppShell, Burger, Container, Group } from '@mantine/core';
import React from 'react';

import classes from './AppContainer.module.css';

import { useDisclosure } from '@mantine/hooks';
import { useState } from 'react';
import Link from 'next/link';

const links = [
  { link: '/', label: 'Home', },
  { link: '/about', label: 'About' },
  { link: 'https://github.com/rjpower/mlwiki', label: 'Github' },
];

function HeaderSimple() {
  const [active, setActive] = useState(links[0].link);

  const items = links.map((link) => (
    <Link
      key={link.label}
      href={link.link}
      className={classes.link}
      data-active={active === link.link || undefined}
      prefetch={false}
    >
      {link.label}
    </Link>
  ));

  return (
    <header className={classes.header}>
      <Container size="md" className={classes.inner}>
        <Group gap="sm">
          {items}
        </Group>
      </Container>
    </header>
  );
}

const AppContainer = ({ children }: { children: React.ReactNode }) => {
  return <AppShell
    padding="md"
    header={{ height: 60 }}
  >
    <AppShell.Header>
      <HeaderSimple />
    </AppShell.Header>

    <AppShell.Main>{children}</AppShell.Main>
  </AppShell>
}

export default AppContainer;

'use client'

import AppContainer from '../components/AppContainer';


import { Button, Container, Group, List, Text, Title } from '@mantine/core';
import Link from 'next/link';
import classes from './page.module.css';

export default function HomePage() {
  return (
    <AppContainer>
      <Container size="md">
        <div className={classes.inner}>
          <div className={classes.content}>
            <Title className={classes.title}>
              MLWiki: a human and ML powered wiki.
            </Title>
            <Text c="dimmed" mt="md">
              MLWiki leverages AI to generate comprenehsive articles on any topic, and combines this
              with careful human editing to improve accuracy and quality.
            </Text>

            <Text c="dimmed" mt="md">
              Check out some of our example articles:
              <List>
                <List.Item>
                  <a href="/view/Quantum_Computing">Quantum Computing</a>
                </List.Item>
                <List.Item>
                  <a href="/view/MHC-I">MHC-1 Antigen Presentation</a>
                </List.Item>
                <List.Item>
                  <a href="/view/Chewbacca_Defense">Chewbacca Defense</a>
                </List.Item>
              </List>
            </Text>

            <Group mt={30}>
              <Button radius="xl" size="md" className={classes.control} component={Link} prefetch={false} href="/random">
                Random Article
              </Button>
              <Button variant="default" radius="xl" size="md" className={classes.control} component={Link} prefetch={false} href="/about">
                About
              </Button>
              <Button variant="default" radius="xl" size="md" className={classes.control} component={Link} prefetch={false} href="https://github.com/rjpower/mlwiki">
                Source code
              </Button>
            </Group>
          </div>
        </div>
      </Container>
    </AppContainer>
  );
}

'use client'

import { Container, Text, Title } from '@mantine/core';
import AppContainer from '../../components/AppContainer';

export default function AboutPage() {
  return (
    <AppContainer>
      <Container size="md">
        <Title>About MLWiki</Title>
        <Text mt="md">
          MLWiki is an experimental wiki powered by large language models (LLMs).
          It aims to generate comprehensive articles on a wide range of topics,
          combining AI-generated content with human editing to ensure accuracy and quality.
        </Text>
      </Container>
    </AppContainer>
  );
}

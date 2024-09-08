'use client'

import { Box, Button, Divider, Grid, GridCol, Pill, Select, Space } from '@mantine/core';
import AppContainer from '../../../components/AppContainer';
import { useState } from 'react';

import { ViewArticle, fetchArticle } from '../../../components/ViewArticle';


export default function Page({ params }: { params: { article: string } }) {
  const [model, setModel] = useState<string | null>('gpt-3.5-turbo');
  return (
    <>
      <title>{`MLWiki - ${params.article}`}</title>
      <AppContainer>
        <Box>
          <Grid>
            <GridCol span="content">
              <Pill>Article</Pill>
              <Divider orientation="vertical" />
            </GridCol>
            <GridCol span="auto">
            </GridCol>
          </Grid>
          <ViewArticle fetcher={(controller: AbortController) =>
            fetchArticle(params.article, model || 'gpt-3.5-turbo', controller)} cacheKey={params.article}></ViewArticle>
        </Box>
      </AppContainer>
    </>
  );
}

type Props = {
  params: { article: string }
}
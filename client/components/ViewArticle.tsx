'use client'

import { Progress, Paper } from '@mantine/core';
import { useEffect, useState } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function ArticleContent({ markdown }: { markdown: string }) {
  return <Paper>
    <Markdown remarkPlugins={[remarkGfm]}>{markdown}</Markdown>
  </Paper>
}

export function readChunks(reader: ReadableStreamDefaultReader) {
  return {
    async*[Symbol.asyncIterator]() {
      let pendingLines = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        let byteArray = value;
        let jsonText = new TextDecoder().decode(byteArray);
        pendingLines = pendingLines.concat(jsonText);
        // Split pendingLines on `\n` and yield each line except the last one.
        let lines = pendingLines.split('\n');
        pendingLines = lines.pop()!;
        for (let line of lines) {
          yield line;
        }
      }
      // Yield the last line if not empty.
      if (pendingLines.length > 0) {
        yield pendingLines;
      }
    },
  };
}

export function fetchPreview(articleConfig: any, controller: AbortController): Promise<Response> {
  return fetch(`/api/article/preview`, {
    method: 'POST',
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(articleConfig),
    signal: controller.signal,
  });
}

export function fetchArticle(article: string, model: string, controller: AbortController): Promise<Response> {
  return fetch(`/api/article/view?article=${article}&model=${model}`, { signal: controller.signal, });
}

export function ViewArticle({ fetcher, cacheKey }: {
  fetcher: (controller: AbortController) => Promise<Response>, cacheKey: any
}) {
  const [articleMarkdown, setMarkdown] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [progress, setProgress] = useState<number>(0);

  useEffect(() => {
    const fetchMarkdown = async () => {
      let controller = new AbortController();
      fetcher(controller)
        .then(async (response) => {
          const reader = response.body!.getReader();
          let combinedMarkdown = '';
          let i = 0;
          for await (const chunk of readChunks(reader)) {
            i += 1;
            const json = JSON.parse(chunk);
            const body = json.content;
            combinedMarkdown = combinedMarkdown.concat(body);
            try {
              setMarkdown(combinedMarkdown);
              setProgress(i < 10 ? i * 10 : 100 - 100 / Math.pow(i, 2));
            } catch (err) {
              setMarkdown(`Error while fetching article: ${err}`);
            }
          }
          setLoading(false);
        })
        .catch((err) => {
          setLoading(false);
          setMarkdown(`Error while fetching article: ${err.status}`);
        })

      return () => controller.abort();
    };

    fetchMarkdown();
  }, [cacheKey, fetcher]);

  let loader = <div />;
  if (loading) {
    loader = (<Paper shadow="xs" p="xl">
      <div>Loading article. This can take up to 10 seconds, please be patient.</div>
      <div>Results should stream, but due to my inability to setup an nginx proxy, they probably won&apos;t.</div>
      <Progress value={progress} />
    </Paper>);
  }

  return (<>
    {loader}
    <ArticleContent markdown={articleMarkdown} />
  </>);

}
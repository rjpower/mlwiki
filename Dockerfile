FROM nikolaik/python-nodejs:latest

EXPOSE 3000

USER pn
WORKDIR /home/pn/app
COPY --chown=pn:pn client/package.json ./client/package.json
RUN cd /home/pn/app/client && npm install
COPY --chown=pn:pn client/ ./client/
ENV NEXT_TELEMETRY_DISABLED=1
RUN cd client && npm run build && rm -rf /app/client/node_modules && rm -rf /app/client/.next/cache
RUN cd client && cp -r .next/static .next/standalone/.next/ && cp -r public .next/standalone/.next

COPY --chown=pn:pn pyproject.toml ./
RUN uv venv && uv sync --no-install-project
COPY --chown=pn:pn api ./api/
COPY --chown=pn:pn run.sh ./run.sh

ENTRYPOINT [ "/home/pn/app/run.sh" ]

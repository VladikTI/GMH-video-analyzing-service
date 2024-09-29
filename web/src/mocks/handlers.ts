import { http, HttpResponse } from 'msw';
import { faker } from '@faker-js/faker';

import details from './details.json';

const data = faker.helpers.multiple(
  () => ({
    id: faker.string.uuid(),
    title: faker.food.dish(),
    tags: faker.helpers.multiple(() => faker.food.adjective()),
  }),
  { count: 10 }
);

export const handlers = [
  http.get('/api/videos', () => {
    // return HttpResponse.json({error: 'Sorry'}, {status: 500});

    return HttpResponse.json(data);
  }),
  http.get('/api/tags', () => {
    const tags = new Set();

    data.forEach((item) => item.tags.forEach((tag) => tags.add(tag)));

    return HttpResponse.json({
      tags: [...tags].map((item) => ({ name: item })),
    });
  }),
  http.get('/api/details', ({request}) => {
    const url = new URL(request.url)
    const id = url.searchParams.get('id')

    return HttpResponse.json({ id, details });
  }),
  http.post('/api/upload', async ({ request }) => {
    const data = await request.formData();
    const file = data.get('file');

    if (!file) {
      return new HttpResponse('Missing document', { status: 400 });
    }

    if (!(file instanceof File)) {
      return new HttpResponse('Uploaded document is not a File', {
        status: 400,
      });
    }

    return HttpResponse.json({
      contents: await file.name,
    });
  }),
];

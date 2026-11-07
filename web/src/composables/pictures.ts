// Pictures decoded before they are shown, by address: an <img> given one of them paints at once. The last `size` are kept;
// one that failed is asked again the next time.

export function pictureCache(size = 128): (url: string) => Promise<boolean> {
  const cache = new Map<string, Promise<boolean>>();
  return (url) => {
    let loaded = cache.get(url);
    if (loaded === undefined) {
      const image = new Image();
      image.decoding = 'async';
      image.src = url;
      loaded = image.decode().then(() => true, () => {
        cache.delete(url);
        return false;
      });
      cache.set(url, loaded);
      if (cache.size > size) cache.delete(cache.keys().next().value as string);
    }
    return loaded;
  };
}

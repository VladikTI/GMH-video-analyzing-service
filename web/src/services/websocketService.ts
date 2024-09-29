interface WebsocketServiceConstructorParams {
  url: string;
  callbacks: {
    onMessage: (data: string) => void;
    onOpen: () => void;
    onClose: () => void;
    onError: () => void;
    onSend?: () => void;
    onDisconnect?: () => void;
  };
}

interface SendOpts {
  type: string;
  data: any;
}

class WebsocketService {
  private socket: WebSocket;
  private onSend?: () => void;
  private onDisconnect?: () => void;

  constructor({ url, callbacks }: WebsocketServiceConstructorParams) {
    this.socket = new WebSocket(url);
    this.onSend = callbacks.onSend;
    this.onDisconnect = callbacks.onDisconnect;

    this.socket.onopen = () => {
      callbacks.onOpen();
    };

    this.socket.onclose = () => {
      callbacks.onClose();
    };

    this.socket.onerror = () => {
      callbacks.onError();
    };

    this.socket.onmessage = (e: MessageEvent) => {
      callbacks.onMessage(e.data);
    };
  }

  public send = (opts: SendOpts) => {
    this.socket.send(JSON.stringify(opts));
    this.onSend?.();
  };

  public disconnect = () => {
    this.socket.close();
    this.onDisconnect?.();
  }
}

export default WebsocketService;

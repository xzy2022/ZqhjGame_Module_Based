// 复用官方网页与 Redis 桥接，仅观测 CLI 会话，不创建第二个仿真或修改官方资源。
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const [runtime, redisHost, redisPort, webPort, readyPath] = process.argv.slice(2);
const {RedisWebSocketBridge} = require(path.join(runtime, 'visualization/dist-bridge/bridge/server.js'));

class ObserverBridge extends RedisWebSocketBridge {
  // 官方前端有实体操作入口；只读观测绝不转发任何 publish 控制消息。
  async handlePublish() {}
}
const bridge = new ObserverBridge({redisHost, redisPort: Number(redisPort),
  wsPort: 0, camHttpPort: 0, camWsPort: 0});
const frontend = path.resolve(runtime, 'frontend');
const mime = {'.html':'text/html; charset=utf-8', '.js':'application/javascript',
  '.css':'text/css', '.json':'application/json', '.png':'image/png',
  '.jpg':'image/jpeg', '.svg':'image/svg+xml', '.woff2':'font/woff2'};
let server;
let ports;
let stopping = false;

function respondJson(res, status, data) {
  res.writeHead(status, {'Content-Type':'application/json; charset=utf-8'});
  res.end(JSON.stringify(data));
}

function serve(req, res) {
  const url = new URL(req.url, 'http://127.0.0.1');
  if (req.method !== 'GET') return respondJson(res, 403, {error:'CLI 实验由终端控制，网页仅供观测'});
  if (url.pathname === '/api/sim/status') {
    return respondJson(res, 200, {status:'running', scenario:'coop_decoy', sessionId:'cli-observer', error:null});
  }
  if (url.pathname === '/api/scenarios') {
    return respondJson(res, 200, {scenarios:[{id:'coop_decoy', name:'命令行实验实时观测',
      description:'运行控制请使用实验终端', available:true, algorithms:[]}]});
  }
  if (url.pathname.startsWith('/api/algorithms/')) return respondJson(res, 200, {algorithms:[]});
  if (url.pathname.startsWith('/api/')) return respondJson(res, 404, {error:'not_found'});
  const filename = path.resolve(frontend, '.' + decodeURIComponent(url.pathname === '/' ? '/index.html' : url.pathname));
  if (!filename.startsWith(frontend + path.sep)) return respondJson(res, 403, {error:'forbidden'});
  fs.readFile(filename, (error, data) => {
    if (error) return respondJson(res, 404, {error:'not_found'});
    const ext = path.extname(filename).toLowerCase();
    if (ext === '.html') {
      // 只在 HTTP 响应里注入端口和观测提示，官方磁盘上的 HTML 保持原样。
      const config = {wsPort:ports.ws, camPort:ports.web, camWsPort:ports.camera,
        camBaseUrl:`http://127.0.0.1:${ports.web}`};
      const injected = `<script>window.__OPENSIM__=${JSON.stringify(config)};</script>
        <style>#scenario-selector{display:none!important}#cli-observer-note{position:fixed;top:52px;left:50%;transform:translateX(-50%);z-index:20000;background:#142333;color:#e6edf6;padding:9px 16px;border:1px solid #38bdf8;border-radius:8px;font:14px sans-serif;pointer-events:none}</style>`;
      data = data.toString('utf8').replace('</head>', injected + '</head>')
        .replace('</body>', '<div id="cli-observer-note">命令行实验 · 只读实时观测 · 运行控制请使用终端</div></body>');
    }
    res.writeHead(200, {'Content-Type':mime[ext] || 'application/octet-stream'});
    res.end(data);
  });
}

async function stop() {
  if (stopping) return;
  stopping = true;
  // 所有退出操作都只针对本进程持有的服务器、套接字和 Redis 连接。
  if (server) server.close();
  for (const client of bridge.clients) client.terminate();
  await bridge.stop();
  process.exit(0);
}
process.stdin.on('data', stop);
process.stdin.on('end', stop);
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
async function main() {
  await bridge.start();
  bridge.wss.on('connection', ws => ws.send(JSON.stringify({type:'session', status:'running',
    scenario:'coop_decoy', sessionId:'cli-observer'})));
  server = http.createServer(serve);
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(Number(webPort), '127.0.0.1', resolve);
  });
  ports = {web:server.address().port, ws:bridge.wss.address().port,
    camera:bridge.cameraWsServer.wss.address().port};
  const info = {url:`http://127.0.0.1:${ports.web}`, ports, redisHost, redisPort:Number(redisPort),
    pid:process.pid, mode:'read_only', runtime};
  fs.writeFileSync(readyPath, JSON.stringify(info, null, 2), 'utf8');
  console.log('CLI_OBSERVER_READY ' + info.url);
}
main().catch(error => {console.error(error); process.exit(1);});

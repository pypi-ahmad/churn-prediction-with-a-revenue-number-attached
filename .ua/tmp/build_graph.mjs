import fs from 'node:fs';
const root=process.cwd();
const scan=JSON.parse(fs.readFileSync('.ua/intermediate/scan-files.json'));
const imp=JSON.parse(fs.readFileSync('.ua/intermediate/imports.json')).importMap;
const structure=JSON.parse(fs.readFileSync('.ua/intermediate/structure.json'));
const fileTypes={code:'file',config:'config',docs:'document',infra:'service',data:'table',script:'file',markup:'file'};
const nodes=[]; const ids=new Set();
for(const f of scan.files){const type=fileTypes[f.fileCategory]||'file'; const id=`${type}:${f.path}`; ids.add(id); nodes.push({id,type,name:f.path.split('/').pop(),filePath:f.path,summary:`${f.fileCategory} file (${f.language}, ${f.sizeLines} lines).`,tags:[f.fileCategory,f.language]});}
for(const r of structure.results){const parent=`${fileTypes[r.fileCategory]||'file'}:${r.path}`; for(const fn of (r.functions||[])){const id=`function:${r.path}:${fn.name}`; nodes.push({id,type:'function',name:fn.name,filePath:r.path,summary:`Function ${fn.name} defined in ${r.path}.`,tags:['function',r.language]}); ids.add(id);}}
const edges=[];
for(const [src,targets] of Object.entries(imp)){const sid=`file:${src}`; for(const t of targets){const tid=`file:${t}`; if(ids.has(sid)&&ids.has(tid)) edges.push({source:sid,target:tid,type:'imports',weight:0.7});}}
for(const n of nodes.filter(n=>n.type==='function')){const parent=`file:${n.filePath}`; if(ids.has(parent)) edges.push({source:parent,target:n.id,type:'contains',weight:1});}
const fileNodes=nodes.filter(n=>['file','config','document','service','table'].includes(n.type));
const layers=[{id:'layer:documentation',name:'Documentation',description:'Project README, contribution guidance, and tutorials.',nodeIds:fileNodes.filter(n=>n.type==='document').map(n=>n.id)},{id:'layer:configuration',name:'Configuration',description:'Project and workflow configuration files.',nodeIds:fileNodes.filter(n=>n.type==='config').map(n=>n.id)},{id:'layer:application',name:'Application and analysis',description:'Python source code and notebooks implementing churn and revenue analysis.',nodeIds:fileNodes.filter(n=>n.type==='file').map(n=>n.id)}].filter(x=>x.nodeIds.length);
const tour=[{order:1,title:'Project Overview',description:'Start with the README to understand the churn and revenue analysis goals.',nodeIds:ids.has('document:README.md')?['document:README.md']:[]},{order:2,title:'Core Python Package',description:'Inspect the source package for metrics, modeling, feature engineering, and value policy logic.',nodeIds:fileNodes.filter(n=>n.filePath.startsWith('src/')).map(n=>n.id)},{order:3,title:'Experiments and Tutorials',description:'Review notebooks and tutorial documents for dataset-specific workflows and methodology.',nodeIds:fileNodes.filter(n=>n.filePath.startsWith('notebooks/')||n.filePath.startsWith('docs/')).map(n=>n.id)}].filter(x=>x.nodeIds.length);
const graph={version:'1.0.0',project:{name:'churn-prediction-with-a-revenue-number-attached',languages:[...new Set(scan.files.map(f=>f.language))].sort(),frameworks:['scikit-learn','CatBoost','XGBoost','LightGBM'],description:'A Python project for churn prediction that attaches customer value and revenue-oriented decision analysis to model outputs.',analyzedAt:new Date().toISOString(),gitCommitHash:process.env.GIT_HASH},nodes,edges,layers,tour};
fs.writeFileSync('.ua/knowledge-graph.json',JSON.stringify(graph,null,2));
fs.writeFileSync('.ua/intermediate/assembled-graph.json',JSON.stringify(graph,null,2));
fs.writeFileSync('.ua/meta.json',JSON.stringify({lastAnalyzedAt:graph.project.analyzedAt,gitCommitHash:process.env.GIT_HASH,version:'1.0.0',analyzedFiles:scan.totalFiles},null,2));
console.log(JSON.stringify({nodes:nodes.length,edges:edges.length,layers:layers.length,tour:tour.length},null,2));
